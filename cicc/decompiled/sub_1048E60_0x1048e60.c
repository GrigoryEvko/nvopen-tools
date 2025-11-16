// Function: sub_1048E60
// Address: 0x1048e60
//
__int64 __fastcall sub_1048E60(__int64 a1, __int64 a2)
{
  int v2; // eax
  char *v3; // rax
  char *v4; // rax
  __int64 *v5; // rax
  __int64 *v6; // r12
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  __int64 *v10; // r13
  __int64 *v11; // r12
  __int64 v12; // r15
  __int64 *v13; // rbx
  __int64 *v14; // r14
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rdi
  __int64 v22; // rdi
  __int64 *v23; // r12
  __int64 *v24; // rbx
  __int64 v25; // rdi
  __int64 *v26; // r12
  __int64 v27; // rsi
  _BYTE *v28; // rbx
  __int64 result; // rax
  _BYTE *v30; // r12
  __int64 v31; // r13
  __int64 v32; // rdi
  unsigned int v33; // ecx
  _QWORD *v34; // rdi
  __int64 v35; // r8
  unsigned int v36; // eax
  int v37; // eax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  int v40; // ebx
  __int64 v41; // r12
  _QWORD *v42; // rax
  _QWORD *i; // rdx
  __int64 *v44; // rax
  __int64 *v45; // rbx
  __int64 *v46; // r13
  __int64 v47; // rdi
  unsigned int v48; // ecx
  __int64 v49; // rsi
  __int64 *v50; // rbx
  __int64 v51; // rdi
  _QWORD *v52; // r8
  __int64 *v53; // [rsp+8h] [rbp-298h]
  _QWORD v54[2]; // [rsp+30h] [rbp-270h] BYREF
  char v55; // [rsp+40h] [rbp-260h] BYREF
  _BYTE *v56; // [rsp+48h] [rbp-258h]
  __int64 v57; // [rsp+50h] [rbp-250h]
  _BYTE v58[56]; // [rsp+58h] [rbp-248h] BYREF
  __int64 v59; // [rsp+90h] [rbp-210h]
  __int64 v60; // [rsp+98h] [rbp-208h]
  char v61; // [rsp+A0h] [rbp-200h]
  int v62; // [rsp+A4h] [rbp-1FCh]
  int v63; // [rsp+A8h] [rbp-1F8h]
  __int64 v64; // [rsp+B0h] [rbp-1F0h] BYREF
  _QWORD *v65; // [rsp+B8h] [rbp-1E8h]
  __int64 v66; // [rsp+C0h] [rbp-1E0h]
  unsigned int v67; // [rsp+C8h] [rbp-1D8h]
  __int64 *v68; // [rsp+D0h] [rbp-1D0h]
  __int64 *v69; // [rsp+D8h] [rbp-1C8h]
  __int64 v70; // [rsp+E0h] [rbp-1C0h]
  __int64 v71; // [rsp+E8h] [rbp-1B8h]
  __int64 j; // [rsp+F0h] [rbp-1B0h]
  __int64 *v73; // [rsp+F8h] [rbp-1A8h]
  __int64 v74; // [rsp+100h] [rbp-1A0h]
  _BYTE v75[32]; // [rsp+108h] [rbp-198h] BYREF
  __int64 *v76; // [rsp+128h] [rbp-178h]
  __int64 v77; // [rsp+130h] [rbp-170h]
  _QWORD v78[3]; // [rsp+138h] [rbp-168h] BYREF
  _QWORD v79[13]; // [rsp+150h] [rbp-150h] BYREF
  char v80; // [rsp+1B8h] [rbp-E8h] BYREF
  _QWORD v81[2]; // [rsp+1F8h] [rbp-A8h] BYREF
  char v82; // [rsp+208h] [rbp-98h] BYREF
  char v83; // [rsp+268h] [rbp-38h] BYREF

  v54[0] = &v55;
  v54[1] = 0x100000000LL;
  v56 = v58;
  v57 = 0x600000000LL;
  v2 = *(_DWORD *)(a2 + 92);
  v60 = a2;
  v63 = v2;
  v59 = 0;
  v61 = 0;
  v62 = 0;
  sub_B1F440((__int64)v54);
  v73 = (__int64 *)v75;
  v74 = 0x400000000LL;
  v76 = v78;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  j = 0;
  v77 = 0;
  v78[0] = 0;
  v78[1] = 1;
  sub_D50CB0((__int64)&v64, (__int64)v54);
  v3 = &v80;
  memset(v79, 0, 96);
  v79[12] = 1;
  do
  {
    *(_QWORD *)v3 = -4096;
    v3 += 16;
  }
  while ( v3 != (char *)v81 );
  v81[0] = 0;
  v4 = &v82;
  v81[1] = 1;
  do
  {
    *(_QWORD *)v4 = -4096;
    v4 += 24;
    *((_DWORD *)v4 - 4) = 0x7FFFFFFF;
  }
  while ( v4 != &v83 );
  sub_FF9360(v79, a2, (__int64)&v64, 0, (__int64)v54, 0);
  v5 = (__int64 *)sub_22077B0(8);
  if ( v5 )
  {
    v53 = v5;
    sub_FE7FB0(v5, (const char *)a2, (__int64)v79, (__int64)&v64);
    v5 = v53;
  }
  v6 = *(__int64 **)(a1 + 16);
  *(_QWORD *)(a1 + 16) = v5;
  if ( v6 )
  {
    sub_FDC110(v6);
    a2 = 8;
    j_j___libc_free_0(v6, 8);
    v5 = *(__int64 **)(a1 + 16);
  }
  *(_QWORD *)(a1 + 8) = v5;
  sub_D77880((__int64)v79);
  ++v64;
  if ( !(_DWORD)v66 )
  {
    if ( !HIDWORD(v66) )
      goto LABEL_15;
    v7 = v67;
    if ( v67 > 0x40 )
    {
      a2 = 16LL * v67;
      sub_C7D6A0((__int64)v65, a2, 8);
      v65 = 0;
      v66 = 0;
      v67 = 0;
      goto LABEL_15;
    }
    goto LABEL_12;
  }
  v33 = 4 * v66;
  a2 = 64;
  v7 = v67;
  if ( (unsigned int)(4 * v66) < 0x40 )
    v33 = 64;
  if ( v67 <= v33 )
  {
LABEL_12:
    v8 = v65;
    v9 = &v65[2 * v7];
    if ( v65 != v9 )
    {
      do
      {
        *v8 = -4096;
        v8 += 2;
      }
      while ( v9 != v8 );
    }
    v66 = 0;
    goto LABEL_15;
  }
  v34 = v65;
  v35 = 2LL * v67;
  if ( (_DWORD)v66 == 1 )
  {
    v41 = 2048;
    v40 = 128;
LABEL_70:
    sub_C7D6A0((__int64)v65, v35 * 8, 8);
    a2 = 8;
    v67 = v40;
    v42 = (_QWORD *)sub_C7D670(v41, 8);
    v66 = 0;
    v65 = v42;
    for ( i = &v42[2 * v67]; i != v42; v42 += 2 )
    {
      if ( v42 )
        *v42 = -4096;
    }
    goto LABEL_15;
  }
  _BitScanReverse(&v36, v66 - 1);
  v37 = 1 << (33 - (v36 ^ 0x1F));
  if ( v37 < 64 )
    v37 = 64;
  if ( v37 != v67 )
  {
    v38 = (4 * v37 / 3u + 1) | ((unsigned __int64)(4 * v37 / 3u + 1) >> 1);
    v39 = ((v38 | (v38 >> 2)) >> 4) | v38 | (v38 >> 2) | ((((v38 | (v38 >> 2)) >> 4) | v38 | (v38 >> 2)) >> 8);
    v40 = (v39 | (v39 >> 16)) + 1;
    v41 = 16 * ((v39 | (v39 >> 16)) + 1);
    goto LABEL_70;
  }
  v66 = 0;
  v52 = &v65[v35];
  do
  {
    if ( v34 )
      *v34 = -4096;
    v34 += 2;
  }
  while ( v52 != v34 );
LABEL_15:
  v10 = v69;
  v11 = v68;
  if ( v68 != v69 )
  {
    do
    {
      v12 = *v11;
      v13 = *(__int64 **)(*v11 + 16);
      if ( *(__int64 **)(*v11 + 8) == v13 )
      {
        *(_BYTE *)(v12 + 152) = 1;
      }
      else
      {
        v14 = *(__int64 **)(*v11 + 8);
        do
        {
          v15 = *v14++;
          sub_D47BB0(v15, a2);
        }
        while ( v13 != v14 );
        *(_BYTE *)(v12 + 152) = 1;
        v16 = *(_QWORD *)(v12 + 8);
        if ( v16 != *(_QWORD *)(v12 + 16) )
          *(_QWORD *)(v12 + 16) = v16;
      }
      v17 = *(_QWORD *)(v12 + 32);
      if ( v17 != *(_QWORD *)(v12 + 40) )
        *(_QWORD *)(v12 + 40) = v17;
      ++*(_QWORD *)(v12 + 56);
      if ( *(_BYTE *)(v12 + 84) )
      {
        *(_QWORD *)v12 = 0;
      }
      else
      {
        v18 = 4 * (*(_DWORD *)(v12 + 76) - *(_DWORD *)(v12 + 80));
        v19 = *(unsigned int *)(v12 + 72);
        if ( v18 < 0x20 )
          v18 = 32;
        if ( (unsigned int)v19 > v18 )
        {
          sub_C8C990(v12 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v12 + 64), -1, 8 * v19);
        }
        v20 = *(_BYTE *)(v12 + 84);
        *(_QWORD *)v12 = 0;
        if ( !v20 )
          _libc_free(*(_QWORD *)(v12 + 64), a2);
      }
      v21 = *(_QWORD *)(v12 + 32);
      if ( v21 )
      {
        a2 = *(_QWORD *)(v12 + 48) - v21;
        j_j___libc_free_0(v21, a2);
      }
      v22 = *(_QWORD *)(v12 + 8);
      if ( v22 )
      {
        a2 = *(_QWORD *)(v12 + 24) - v22;
        j_j___libc_free_0(v22, a2);
      }
      ++v11;
    }
    while ( v10 != v11 );
    if ( v68 != v69 )
      v69 = v68;
  }
  v23 = v76;
  v24 = &v76[2 * (unsigned int)v77];
  if ( v76 != v24 )
  {
    do
    {
      a2 = v23[1];
      v25 = *v23;
      v23 += 2;
      sub_C7D6A0(v25, a2, 16);
    }
    while ( v24 != v23 );
  }
  LODWORD(v77) = 0;
  if ( !(_DWORD)v74 )
    goto LABEL_40;
  v44 = v73;
  v78[0] = 0;
  v45 = &v73[(unsigned int)v74];
  v46 = v73 + 1;
  v71 = *v73;
  for ( j = v71 + 4096; v45 != v46; v44 = v73 )
  {
    v47 = *v46;
    v48 = (unsigned int)(v46 - v44) >> 7;
    v49 = 4096LL << v48;
    if ( v48 >= 0x1E )
      v49 = 0x40000000000LL;
    ++v46;
    sub_C7D6A0(v47, v49, 16);
  }
  LODWORD(v74) = 1;
  a2 = 4096;
  sub_C7D6A0(*v44, 4096, 16);
  v50 = v76;
  v26 = &v76[2 * (unsigned int)v77];
  if ( v76 != v26 )
  {
    do
    {
      a2 = v50[1];
      v51 = *v50;
      v50 += 2;
      sub_C7D6A0(v51, a2, 16);
    }
    while ( v26 != v50 );
LABEL_40:
    v26 = v76;
  }
  if ( v26 != v78 )
    _libc_free(v26, a2);
  if ( v73 != (__int64 *)v75 )
    _libc_free(v73, a2);
  if ( v68 )
    j_j___libc_free_0(v68, v70 - (_QWORD)v68);
  v27 = 16LL * v67;
  sub_C7D6A0((__int64)v65, v27, 8);
  v28 = v56;
  result = (unsigned int)v57;
  v30 = &v56[8 * (unsigned int)v57];
  if ( v56 != v30 )
  {
    do
    {
      v31 = *((_QWORD *)v30 - 1);
      v30 -= 8;
      if ( v31 )
      {
        v32 = *(_QWORD *)(v31 + 24);
        if ( v32 != v31 + 40 )
          _libc_free(v32, v27);
        v27 = 80;
        result = j_j___libc_free_0(v31, 80);
      }
    }
    while ( v28 != v30 );
    v30 = v56;
  }
  if ( v30 != v58 )
    result = _libc_free(v30, v27);
  if ( (char *)v54[0] != &v55 )
    return _libc_free(v54[0], v27);
  return result;
}
