// Function: sub_191E180
// Address: 0x191e180
//
__int64 __fastcall sub_191E180(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  int v11; // eax
  int v13; // edx
  __int64 v14; // rsi
  unsigned int v15; // eax
  __int64 v16; // rcx
  int v18; // edi
  __int64 *v19; // rsi
  unsigned int v20; // eax
  unsigned int v21; // eax
  unsigned int v22; // ecx
  __int64 v23; // rdx
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *i; // rdx
  _QWORD *v27; // r15
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rsi
  int v31; // edx
  _QWORD *v32; // r12
  int v33; // ecx
  __int64 v34; // rsi
  unsigned int v35; // edi
  __int64 *v36; // rdx
  __int64 v37; // r8
  _QWORD *v38; // rbx
  _QWORD *v39; // r13
  _QWORD *v40; // r15
  __int64 v41; // r12
  int v42; // eax
  int v43; // eax
  __int64 v44; // rsi
  int v45; // edi
  unsigned int v46; // ecx
  __int64 *v47; // rbx
  __int64 v48; // rdx
  __int64 v49; // r13
  int v50; // edx
  unsigned int v51; // eax
  unsigned int v52; // ebx
  char v53; // al
  __int64 v54; // rdi
  __int64 v55; // rax
  bool v56; // zf
  _QWORD *v57; // rax
  __int64 v58; // rdx
  _QWORD *j; // rdx
  int v60; // r9d
  _QWORD *v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rdx
  _QWORD *v64; // [rsp+0h] [rbp-70h]
  _QWORD *v65; // [rsp+8h] [rbp-68h]
  unsigned __int64 v66; // [rsp+10h] [rbp-60h]
  unsigned int v68; // [rsp+28h] [rbp-48h]
  char v69; // [rsp+2Eh] [rbp-42h]
  unsigned __int8 v70; // [rsp+2Fh] [rbp-41h]
  __int64 v71; // [rsp+30h] [rbp-40h] BYREF
  char v72; // [rsp+38h] [rbp-38h]

  v11 = *(_DWORD *)(a1 + 72);
  if ( v11 )
  {
    v13 = v11 - 1;
    v14 = *(_QWORD *)(a1 + 56);
    v15 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = *(_QWORD *)(v14 + 8LL * v15);
    if ( a2 == v16 )
      return 0;
    v18 = 1;
    while ( v16 != -8 )
    {
      v15 = v13 & (v18 + v15);
      v16 = *(_QWORD *)(v14 + 8LL * v15);
      if ( a2 == v16 )
        return 0;
      ++v18;
    }
  }
  v19 = *(__int64 **)(a1 + 8);
  if ( v19 )
  {
    sub_1368C40((__int64)&v71, v19, a2);
    if ( !v72 || !v71 )
      return 0;
  }
  v20 = *(_DWORD *)(a1 + 520);
  ++*(_QWORD *)(a1 + 512);
  v21 = v20 >> 1;
  if ( v21 )
  {
    if ( (*(_BYTE *)(a1 + 520) & 1) == 0 )
    {
      v22 = 4 * v21;
      goto LABEL_13;
    }
LABEL_32:
    v24 = (_QWORD *)(a1 + 528);
    v25 = 8;
    goto LABEL_16;
  }
  if ( !*(_DWORD *)(a1 + 524) )
    goto LABEL_19;
  v22 = 0;
  if ( (*(_BYTE *)(a1 + 520) & 1) != 0 )
    goto LABEL_32;
LABEL_13:
  v23 = *(unsigned int *)(a1 + 536);
  if ( (unsigned int)v23 <= v22 || (unsigned int)v23 <= 0x40 )
  {
    v24 = *(_QWORD **)(a1 + 528);
    v25 = 2 * v23;
LABEL_16:
    for ( i = &v24[v25]; i != v24; v24 += 2 )
      *v24 = -8;
    *(_QWORD *)(a1 + 520) &= 1uLL;
    goto LABEL_19;
  }
  if ( !v21 || (v51 = v21 - 1) == 0 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 528));
    *(_BYTE *)(a1 + 520) |= 1u;
LABEL_72:
    v56 = (*(_QWORD *)(a1 + 520) & 1LL) == 0;
    *(_QWORD *)(a1 + 520) &= 1uLL;
    if ( v56 )
    {
      v57 = *(_QWORD **)(a1 + 528);
      v58 = 2LL * *(unsigned int *)(a1 + 536);
    }
    else
    {
      v57 = (_QWORD *)(a1 + 528);
      v58 = 8;
    }
    for ( j = &v57[v58]; j != v57; v57 += 2 )
    {
      if ( v57 )
        *v57 = -8;
    }
    goto LABEL_19;
  }
  _BitScanReverse(&v51, v51);
  v52 = 1 << (33 - (v51 ^ 0x1F));
  if ( v52 - 5 <= 0x3A )
  {
    v52 = 64;
    j___libc_free_0(*(_QWORD *)(a1 + 528));
    v53 = *(_BYTE *)(a1 + 520);
    v54 = 1024;
LABEL_71:
    *(_BYTE *)(a1 + 520) = v53 & 0xFE;
    v55 = sub_22077B0(v54);
    *(_DWORD *)(a1 + 536) = v52;
    *(_QWORD *)(a1 + 528) = v55;
    goto LABEL_72;
  }
  if ( (_DWORD)v23 != v52 )
  {
    j___libc_free_0(*(_QWORD *)(a1 + 528));
    v53 = *(_BYTE *)(a1 + 520) | 1;
    *(_BYTE *)(a1 + 520) = v53;
    if ( v52 <= 4 )
      goto LABEL_72;
    v54 = 16LL * v52;
    goto LABEL_71;
  }
  v56 = (*(_QWORD *)(a1 + 520) & 1LL) == 0;
  *(_QWORD *)(a1 + 520) &= 1uLL;
  if ( v56 )
  {
    v61 = *(_QWORD **)(a1 + 528);
    v62 = 2 * v23;
  }
  else
  {
    v61 = (_QWORD *)(a1 + 528);
    v62 = 8;
  }
  v63 = &v61[v62];
  do
  {
    if ( v61 )
      *v61 = -8;
    v61 += 2;
  }
  while ( v63 != v61 );
LABEL_19:
  *(_DWORD *)(a1 + 600) = 0;
  v27 = *(_QWORD **)(a2 + 48);
  if ( v27 == (_QWORD *)(a2 + 40) )
    return 0;
  v70 = 0;
  v68 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  while ( 1 )
  {
    v28 = (__int64)(v27 - 3);
    if ( !v27 )
      v28 = 0;
    v70 |= sub_191D810(a1, v28, a3, a4, a5, a6, a7, a8, a9, a10);
    v29 = *(unsigned int *)(a1 + 680);
    if ( (_DWORD)v29 )
    {
      v66 = (unsigned __int64)v27;
      v65 = *(_QWORD **)(a2 + 48);
      if ( v65 != v27 )
        v66 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = *(_DWORD *)(a1 + 136);
      v32 = 0;
      if ( v31 )
      {
        v33 = v31 - 1;
        v34 = *(_QWORD *)(a1 + 120);
        v35 = (v31 - 1) & v68;
        v36 = (__int64 *)(v34 + 16LL * v35);
        v37 = *v36;
        if ( a2 == *v36 )
        {
LABEL_37:
          v32 = (_QWORD *)v36[1];
        }
        else
        {
          v50 = 1;
          while ( v37 != -8 )
          {
            v60 = v50 + 1;
            v35 = v33 & (v50 + v35);
            v36 = (__int64 *)(v34 + 16LL * v35);
            v37 = *v36;
            if ( a2 == *v36 )
              goto LABEL_37;
            v50 = v60;
          }
          v32 = 0;
        }
      }
      v38 = *(_QWORD **)(a1 + 672);
      v69 = 0;
      v64 = v27;
      v39 = &v38[v29];
      do
      {
        v40 = (_QWORD *)*v38;
        sub_1AEAA40(*v38);
        if ( *(_QWORD *)a1 )
          sub_14191F0(*(_QWORD *)a1, (__int64)v40);
        if ( v40 == v32 )
        {
          v69 = 1;
          v32 = 0;
        }
        ++v38;
        sub_15F20C0(v40);
      }
      while ( v39 != v38 );
      v41 = *(_QWORD *)(a1 + 144);
      v42 = *(_DWORD *)(v41 + 24);
      if ( v42 )
      {
        v43 = v42 - 1;
        v44 = *(_QWORD *)(v41 + 8);
        v45 = 1;
        v46 = v43 & v68;
        v47 = (__int64 *)(v44 + 16LL * (v43 & v68));
        v48 = *v47;
        if ( a2 == *v47 )
        {
LABEL_46:
          v49 = v47[1];
          if ( v49 )
          {
            if ( (*(_BYTE *)(v49 + 8) & 1) == 0 )
              j___libc_free_0(*(_QWORD *)(v49 + 16));
            j_j___libc_free_0(v49, 552);
          }
          *v47 = -16;
          --*(_DWORD *)(v41 + 16);
          ++*(_DWORD *)(v41 + 20);
        }
        else
        {
          while ( v48 != -8 )
          {
            v46 = v43 & (v45 + v46);
            v47 = (__int64 *)(v44 + 16LL * v46);
            v48 = *v47;
            if ( a2 == *v47 )
              goto LABEL_46;
            ++v45;
          }
        }
      }
      *(_DWORD *)(a1 + 680) = 0;
      if ( v69 )
        sub_1918240(a1, a2);
      v27 = v65 == v64 ? *(_QWORD **)(a2 + 48) : *(_QWORD **)(v66 + 8);
    }
    else
    {
      v27 = (_QWORD *)v27[1];
    }
    if ( (_QWORD *)(a2 + 40) == v27 )
      return v70;
    if ( *(_DWORD *)(a1 + 600) )
    {
      v30 = (__int64)(v27 - 3);
      if ( !v27 )
        v30 = 0;
      v70 |= sub_190B340(a1, v30);
    }
  }
}
