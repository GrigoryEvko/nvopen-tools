// Function: sub_29118F0
// Address: 0x29118f0
//
__int64 __fastcall sub_29118F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rbx
  int v11; // r10d
  unsigned int i; // eax
  __int64 v13; // rdi
  unsigned int v14; // eax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r14
  __int64 v21; // rbx
  __int64 v22; // rsi
  char v23; // bl
  __int64 v24; // r15
  __int64 v25; // r13
  char v26; // al
  __int64 v27; // r14
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v32; // r13
  __int64 *v33; // rbx
  __int64 *v34; // r15
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r13
  __int64 v39; // rsi
  char v40; // al
  unsigned __int64 v41; // r13
  __int64 *v42; // r15
  _QWORD *v43; // rsi
  __int64 v44; // r8
  __int64 v45; // r9
  void *v46; // rsi
  __int64 v47; // rcx
  void *v48; // r12
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  _QWORD *v54; // rbx
  _QWORD *v55; // r14
  void (__fastcall *v56)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v57; // rax
  __int64 v59; // rdx
  __int64 v60; // rcx
  __int64 v61; // [rsp+10h] [rbp-4A0h]
  char v62; // [rsp+10h] [rbp-4A0h]
  __int64 v64; // [rsp+38h] [rbp-478h]
  char v65[8]; // [rsp+40h] [rbp-470h] BYREF
  __int64 j; // [rsp+48h] [rbp-468h] BYREF
  __int64 *v67; // [rsp+50h] [rbp-460h] BYREF
  __int64 v68; // [rsp+58h] [rbp-458h]
  _BYTE v69[64]; // [rsp+60h] [rbp-450h] BYREF
  __int64 v70; // [rsp+A0h] [rbp-410h] BYREF
  _QWORD *v71; // [rsp+A8h] [rbp-408h]
  __int64 v72; // [rsp+B0h] [rbp-400h]
  __int64 (__fastcall *v73)(__int64); // [rsp+B8h] [rbp-3F8h]
  _QWORD v74[2]; // [rsp+C0h] [rbp-3F0h] BYREF
  __int64 v75; // [rsp+D0h] [rbp-3E0h] BYREF
  _BYTE *v76; // [rsp+D8h] [rbp-3D8h]
  __int64 v77; // [rsp+E0h] [rbp-3D0h]
  int v78; // [rsp+E8h] [rbp-3C8h]
  char v79; // [rsp+ECh] [rbp-3C4h]
  _BYTE v80[208]; // [rsp+F0h] [rbp-3C0h] BYREF
  unsigned __int64 v81[2]; // [rsp+1C0h] [rbp-2F0h] BYREF
  _BYTE v82[512]; // [rsp+1D0h] [rbp-2E0h] BYREF
  __int64 v83; // [rsp+3D0h] [rbp-E0h]
  __int64 v84; // [rsp+3D8h] [rbp-D8h]
  __int64 v85; // [rsp+3E0h] [rbp-D0h]
  __int64 v86; // [rsp+3E8h] [rbp-C8h]
  char v87; // [rsp+3F0h] [rbp-C0h]
  __int64 v88; // [rsp+3F8h] [rbp-B8h]
  char *v89; // [rsp+400h] [rbp-B0h]
  __int64 v90; // [rsp+408h] [rbp-A8h]
  int v91; // [rsp+410h] [rbp-A0h]
  char v92; // [rsp+414h] [rbp-9Ch]
  char v93; // [rsp+418h] [rbp-98h] BYREF
  __int16 v94; // [rsp+458h] [rbp-58h]
  _QWORD *v95; // [rsp+460h] [rbp-50h]
  _QWORD *v96; // [rsp+468h] [rbp-48h]
  __int64 v97; // [rsp+470h] [rbp-40h]

  v6 = sub_B2BEC0(a3);
  v7 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v8 = *(unsigned int *)(a4 + 88);
  v9 = *(_QWORD *)(a4 + 72);
  v10 = v7 + 8;
  if ( !(_DWORD)v8 )
    goto LABEL_73;
  v11 = 1;
  for ( i = (v8 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v8 - 1) & v14 )
  {
    v13 = v9 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F81450 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_73;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v9 + 24 * v8 )
  {
LABEL_73:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8;
  }
  v85 = v15;
  v81[0] = (unsigned __int64)v82;
  v89 = &v93;
  v81[1] = 0x1000000000LL;
  v83 = 0;
  v84 = 0;
  v86 = 0;
  v87 = 1;
  v88 = 0;
  v90 = 8;
  v91 = 0;
  v92 = 1;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v16 = sub_B2BE50(a3);
  v70 = v10;
  v73 = sub_29118B0;
  v72 = (__int64)sub_29118C0;
  sub_2A63AD0(v65, v6, &v70, v16);
  if ( v72 )
    ((void (__fastcall *)(__int64 *, __int64 *, __int64))v72)(&v70, &v70, 3);
  if ( (unsigned __int8)sub_310F860(a3) )
    sub_2A73720(v65, a3);
  v17 = *(_QWORD *)(a3 + 80);
  if ( v17 )
    v17 -= 24;
  sub_2A63F40(v65, v17);
  if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
  {
    sub_B2C6D0(a3, v17, v18, v19);
    v20 = *(_QWORD *)(a3 + 96);
    v21 = v20 + 40LL * *(_QWORD *)(a3 + 104);
    if ( (*(_BYTE *)(a3 + 2) & 1) != 0 )
    {
      sub_B2C6D0(a3, v17, v59, v60);
      v20 = *(_QWORD *)(a3 + 96);
    }
  }
  else
  {
    v20 = *(_QWORD *)(a3 + 96);
    v21 = v20 + 40LL * *(_QWORD *)(a3 + 104);
  }
  while ( v21 != v20 )
  {
    v22 = v20;
    v20 += 40;
    sub_2A6B770(v65, v22);
  }
  do
    sub_2A72B90(v65);
  while ( (unsigned __int8)sub_2A6CE80(v65, a3) );
  v23 = 0;
  v24 = *(_QWORD *)(a3 + 80);
  v70 = 0;
  v71 = v74;
  v67 = (__int64 *)v69;
  v68 = 0x800000000LL;
  v72 = 32;
  LODWORD(v73) = 0;
  BYTE4(v73) = 1;
  v64 = a3 + 72;
  if ( v24 == a3 + 72 )
  {
    j = 0;
  }
  else
  {
    v61 = a3;
    v25 = v24;
    do
    {
      while ( 1 )
      {
        v27 = 0;
        if ( v25 )
          v27 = v25 - 24;
        if ( !(unsigned __int8)sub_2A64220(v65, v27) )
          break;
        v26 = sub_2A66DA0(v65, v27, &v70, &unk_5005389, &unk_5005388);
        v25 = *(_QWORD *)(v25 + 8);
        v23 |= v26;
        if ( v64 == v25 )
          goto LABEL_30;
      }
      v30 = (unsigned int)v68;
      v31 = (unsigned int)v68 + 1LL;
      if ( v31 > HIDWORD(v68) )
      {
        sub_C8D5F0((__int64)&v67, v69, v31, 8u, v28, v29);
        v30 = (unsigned int)v68;
      }
      v23 = 1;
      v67[v30] = v27;
      LODWORD(v68) = v68 + 1;
      v25 = *(_QWORD *)(v25 + 8);
    }
    while ( v64 != v25 );
LABEL_30:
    v32 = v61;
    if ( v67 == &v67[(unsigned int)v68] )
    {
      v38 = *(_QWORD *)(v61 + 80);
    }
    else
    {
      v62 = v23;
      v33 = v67;
      v34 = &v67[(unsigned int)v68];
      do
      {
        v35 = sub_AA4FF0(*v33);
        if ( v35 )
          v35 -= 24;
        ++v33;
        sub_F55BE0(v35, 0, (__int64)v81, 0, v36, v37);
      }
      while ( v34 != v33 );
      v23 = v62;
      v38 = *(_QWORD *)(v32 + 80);
    }
    for ( j = 0; v64 != v38; v23 |= v40 )
    {
      v39 = v38 - 24;
      if ( !v38 )
        v39 = 0;
      v40 = sub_2A64290(v65, v39, v81, &j);
      v38 = *(_QWORD *)(v38 + 8);
    }
  }
  v41 = (unsigned __int64)v67;
  v42 = &v67[(unsigned int)v68];
  if ( v67 != v42 )
  {
    do
    {
      while ( 1 )
      {
        v43 = *(_QWORD **)v41;
        if ( (*(_WORD *)(*(_QWORD *)v41 + 2LL) & 0x7FFF) == 0 )
          break;
        v41 += 8LL;
        if ( v42 == (__int64 *)v41 )
          goto LABEL_45;
      }
      v41 += 8LL;
      sub_FFBF00((__int64)v81, v43);
    }
    while ( v42 != (__int64 *)v41 );
  }
LABEL_45:
  sub_2A654F0(v65);
  if ( v67 != (__int64 *)v69 )
    _libc_free((unsigned __int64)v67);
  if ( !BYTE4(v73) )
    _libc_free((unsigned __int64)v71);
  sub_2A665E0(v65);
  v46 = (void *)(a1 + 32);
  v47 = a1;
  v48 = (void *)(a1 + 80);
  if ( v23 )
  {
    v72 = 0x100000002LL;
    v71 = v74;
    v74[0] = &unk_4F81450;
    LODWORD(v73) = 0;
    BYTE4(v73) = 1;
    v75 = 0;
    v76 = v80;
    v77 = 2;
    v78 = 0;
    v79 = 1;
    v70 = 1;
    sub_C8CF70(a1, v46, 2, (__int64)v74, (__int64)&v70);
    v46 = (void *)(a1 + 80);
    sub_C8CF70(a1 + 48, v48, 2, (__int64)v80, (__int64)&v75);
    if ( !v79 )
      _libc_free((unsigned __int64)v76);
    if ( !BYTE4(v73) )
      _libc_free((unsigned __int64)v71);
  }
  else
  {
    v49 = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v46;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v48;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
  }
  sub_FFCE90((__int64)v81, (__int64)v46, v49, v47, v44, v45);
  sub_FFD870((__int64)v81, (__int64)v46, v50, v51, v52, v53);
  sub_FFBC40((__int64)v81, (__int64)v46);
  v54 = v96;
  v55 = v95;
  if ( v96 != v95 )
  {
    do
    {
      v56 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v55[7];
      *v55 = &unk_49E5048;
      if ( v56 )
        v56(v55 + 5, v55 + 5, 3);
      *v55 = &unk_49DB368;
      v57 = v55[3];
      if ( v57 != -4096 && v57 != 0 && v57 != -8192 )
        sub_BD60C0(v55 + 1);
      v55 += 9;
    }
    while ( v54 != v55 );
    v55 = v95;
  }
  if ( v55 )
    j_j___libc_free_0((unsigned __int64)v55);
  if ( !v92 )
    _libc_free((unsigned __int64)v89);
  if ( (_BYTE *)v81[0] != v82 )
    _libc_free(v81[0]);
  return a1;
}
