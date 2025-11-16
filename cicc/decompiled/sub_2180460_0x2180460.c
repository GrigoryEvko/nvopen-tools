// Function: sub_2180460
// Address: 0x2180460
//
__int64 __fastcall sub_2180460(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdi
  unsigned int *v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  unsigned int v17; // ecx
  int *v18; // rdi
  int v19; // esi
  _DWORD *v20; // rax
  int v21; // r10d
  __int64 v22; // r11
  unsigned int v23; // r12d
  int v24; // r14d
  unsigned int v25; // r13d
  unsigned __int64 v26; // rax
  unsigned int v27; // esi
  unsigned int v28; // r9d
  unsigned int v29; // ecx
  __int64 v30; // rdx
  unsigned int *v31; // rax
  unsigned int v32; // edi
  unsigned int *v33; // rdx
  unsigned int v34; // eax
  unsigned int v35; // esi
  __int64 v36; // rcx
  _QWORD *v37; // rax
  int v38; // edi
  unsigned int v39; // esi
  char v40; // r8
  __int64 v41; // rax
  unsigned __int64 v42; // r13
  unsigned __int64 i; // r12
  int *v44; // rsi
  int v46; // ecx
  unsigned int v47; // esi
  int v48; // edx
  __int64 v49; // rdx
  int v50; // r10d
  unsigned int v52; // [rsp+3Ch] [rbp-124h]
  unsigned int *v53; // [rsp+40h] [rbp-120h]
  __int64 v54; // [rsp+48h] [rbp-118h] BYREF
  int v55; // [rsp+54h] [rbp-10Ch] BYREF
  __int64 v56; // [rsp+58h] [rbp-108h] BYREF
  __int64 v57; // [rsp+60h] [rbp-100h] BYREF
  __int64 v58; // [rsp+68h] [rbp-F8h]
  __int64 v59; // [rsp+70h] [rbp-F0h]
  __int64 v60; // [rsp+78h] [rbp-E8h]
  __int64 v61; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v62; // [rsp+88h] [rbp-D8h]
  __int64 v63; // [rsp+90h] [rbp-D0h]
  __int64 v64; // [rsp+98h] [rbp-C8h]
  _BYTE *v65; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+A8h] [rbp-B8h]
  _BYTE v67[16]; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int *v68; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+C8h] [rbp-98h]
  _BYTE v70[16]; // [rsp+D0h] [rbp-90h] BYREF
  _BYTE *v71; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v72; // [rsp+E8h] [rbp-78h]
  _BYTE v73[16]; // [rsp+F0h] [rbp-70h] BYREF
  _BYTE *v74; // [rsp+100h] [rbp-60h] BYREF
  __int64 v75; // [rsp+108h] [rbp-58h]
  _BYTE v76[16]; // [rsp+110h] [rbp-50h] BYREF
  char v77; // [rsp+120h] [rbp-40h]

  v68 = (unsigned int *)v70;
  v6 = *(_DWORD *)(a2 + 40);
  v54 = a2;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = v67;
  v66 = 0x400000000LL;
  v69 = 0x400000000LL;
  v52 = v6;
  if ( !v6 )
  {
    v72 = 0x400000000LL;
    v71 = v73;
    v74 = v76;
    v75 = 0x400000000LL;
    goto LABEL_49;
  }
  v7 = a2;
  v8 = 0;
  v9 = 40LL * v6;
  while ( 1 )
  {
    v10 = v8 + *(_QWORD *)(v7 + 32);
    if ( *(_BYTE *)v10 )
      goto LABEL_5;
    LODWORD(v71) = *(_DWORD *)(v10 + 8);
    if ( (*(_BYTE *)(v10 + 3) & 0x10) != 0 )
      break;
    sub_217F7B0((__int64)&v74, (__int64)&v61, (int *)&v71);
    if ( v77 )
      sub_1525B90((__int64)&v68, &v71);
LABEL_5:
    v8 += 40;
    if ( v9 == v8 )
      goto LABEL_11;
LABEL_6:
    v7 = v54;
  }
  sub_217F7B0((__int64)&v74, (__int64)&v57, (int *)&v71);
  if ( !v77 )
    goto LABEL_5;
  v8 += 40;
  sub_1525B90((__int64)&v65, &v71);
  if ( v9 != v8 )
    goto LABEL_6;
LABEL_11:
  v11 = v68;
  v74 = v76;
  v71 = v73;
  v53 = &v68[(unsigned int)v69];
  v72 = 0x400000000LL;
  v75 = 0x400000000LL;
  if ( v68 == v53 )
  {
    v52 = 0;
    goto LABEL_49;
  }
  v52 = 0;
  while ( 2 )
  {
    v12 = *v11;
    v13 = *(_QWORD *)(a1 + 80);
    v55 = v12;
    if ( (int)v12 < 0 )
    {
      v36 = *(_QWORD *)(a1 + 72);
      v37 = (_QWORD *)(*(_QWORD *)(v13 + 24) + 16 * (v12 & 0x7FFFFFFF));
      LODWORD(v56) = (*(_DWORD *)(*(_QWORD *)(v36 + 280)
                                + 24LL
                                * (*(unsigned __int16 *)(*(_QWORD *)(*v37 & 0xFFFFFFFFFFFFFFF8LL) + 24LL)
                                 + *(_DWORD *)(v36 + 288)
                                 * (unsigned int)((__int64)(*(_QWORD *)(v36 + 264) - *(_QWORD *)(v36 + 256)) >> 3))) > 0x20u)
                   + 1;
      v14 = v37[1];
    }
    else
    {
      LODWORD(v56) = 1;
      v14 = *(_QWORD *)(*(_QWORD *)(v13 + 272) + 8 * v12);
    }
    if ( v14 )
    {
      if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
        goto LABEL_19;
      while ( 1 )
      {
        v14 = *(_QWORD *)(v14 + 32);
        if ( !v14 )
          break;
        if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
        {
LABEL_19:
          while ( 1 )
          {
            v14 = *(_QWORD *)(v14 + 32);
            if ( !v14 )
              goto LABEL_20;
            if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
              goto LABEL_39;
          }
        }
      }
    }
LABEL_39:
    v52 += v56;
LABEL_20:
    sub_1525B90((__int64)&v74, &v56);
    v15 = *(unsigned int *)(a3 + 24);
    if ( (_DWORD)v15 )
    {
      v16 = *(_QWORD *)(a3 + 8);
      v17 = (v15 - 1) & (37 * v55);
      v18 = (int *)(v16 + 8LL * v17);
      v19 = *v18;
      if ( v55 == *v18 )
      {
LABEL_22:
        if ( v18 != (int *)(v16 + 8 * v15) )
        {
          v20 = sub_1E49390(a3, &v55);
          sub_1525B90((__int64)&v71, v20 + 1);
          goto LABEL_24;
        }
      }
      else
      {
        v38 = 1;
        while ( v19 != -1 )
        {
          v50 = v38 + 1;
          v17 = (v15 - 1) & (v38 + v17);
          v18 = (int *)(v16 + 8LL * v17);
          v19 = *v18;
          if ( v55 == *v18 )
            goto LABEL_22;
          v38 = v50;
        }
      }
    }
    sub_1525B90((__int64)&v71, &v56);
LABEL_24:
    if ( ++v11 != v53 )
      continue;
    break;
  }
  v21 = v72;
  if ( !(_DWORD)v72 )
    goto LABEL_49;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
LABEL_27:
  v26 = (unsigned __int64)v71;
  ++v23;
  v27 = *(_DWORD *)&v71[v22];
  v28 = *(_DWORD *)&v74[v22];
  if ( v23 != v21 )
  {
    v29 = v23;
    while ( 1 )
    {
      v30 = 4LL * v29;
      v31 = (unsigned int *)(v30 + v26);
      v32 = *v31;
      if ( *v31 > v27 || v32 == v27 && *(_DWORD *)&v74[4 * v29] < v28 )
      {
        *v31 = v27;
        v33 = (unsigned int *)&v74[v30];
        v27 = v32;
        v34 = *v33;
        *v33 = v28;
        v28 = v34;
        if ( ++v29 == v21 )
        {
LABEL_34:
          *(_DWORD *)&v74[v22] = v28;
          *(_DWORD *)&v71[v22] = v27;
          v35 = v24 + v27;
          if ( v25 < v35 )
            v25 = v35;
          v24 += v28;
          v22 += 4;
          goto LABEL_27;
        }
      }
      else if ( ++v29 == v21 )
      {
        goto LABEL_34;
      }
      v26 = (unsigned __int64)v71;
    }
  }
  v39 = v24 + v27;
  if ( v39 < v25 )
    v39 = v25;
  v52 += v39;
LABEL_49:
  v40 = sub_1FD4240(a4, &v54, &v56);
  v41 = v56;
  if ( !v40 )
  {
    v46 = *(_DWORD *)(a4 + 16);
    v47 = *(_DWORD *)(a4 + 24);
    ++*(_QWORD *)a4;
    v48 = v46 + 1;
    if ( 4 * (v46 + 1) >= 3 * v47 )
    {
      v47 *= 2;
    }
    else if ( v47 - *(_DWORD *)(a4 + 20) - v48 > v47 >> 3 )
    {
LABEL_63:
      *(_DWORD *)(a4 + 16) = v48;
      if ( *(_QWORD *)v41 != -8 )
        --*(_DWORD *)(a4 + 20);
      v49 = v54;
      *(_DWORD *)(v41 + 8) = 0;
      *(_QWORD *)v41 = v49;
      goto LABEL_50;
    }
    sub_1DC6D40(a4, v47);
    sub_1FD4240(a4, &v54, &v56);
    v41 = v56;
    v48 = *(_DWORD *)(a4 + 16) + 1;
    goto LABEL_63;
  }
LABEL_50:
  v42 = (unsigned __int64)v65;
  *(_DWORD *)(v41 + 8) = v52;
  for ( i = v42 + 4LL * (unsigned int)v66; v42 != i; sub_1E49390(a3, v44)[1] = v52 )
  {
    v44 = (int *)v42;
    v42 += 4LL;
  }
  if ( v74 != v76 )
    _libc_free((unsigned __int64)v74);
  if ( v71 != v73 )
    _libc_free((unsigned __int64)v71);
  if ( v68 != (unsigned int *)v70 )
    _libc_free((unsigned __int64)v68);
  if ( v65 != v67 )
    _libc_free((unsigned __int64)v65);
  j___libc_free_0(v62);
  j___libc_free_0(v58);
  return v52;
}
