// Function: sub_2FCB7C0
// Address: 0x2fcb7c0
//
__int64 __fastcall sub_2FCB7C0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v8; // rax
  void *v9; // rcx
  __int64 v10; // r8
  void *v11; // r9
  __int64 v12; // rdx
  __int64 v13; // rsi
  _BYTE *v14; // r14
  int v15; // r10d
  unsigned int i; // eax
  __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned __int64 *v19; // rdx
  void *v20; // r10
  __int64 v21; // r15
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  _QWORD *v26; // rbx
  _QWORD *v27; // r15
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v29; // rax
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  void *v37; // [rsp+10h] [rbp-360h]
  unsigned __int64 *v38; // [rsp+10h] [rbp-360h]
  __int64 v39; // [rsp+20h] [rbp-350h] BYREF
  _BYTE *v40; // [rsp+28h] [rbp-348h]
  __int64 v41; // [rsp+30h] [rbp-340h]
  int v42; // [rsp+38h] [rbp-338h]
  char v43; // [rsp+3Ch] [rbp-334h]
  _BYTE v44[16]; // [rsp+40h] [rbp-330h] BYREF
  __int64 v45; // [rsp+50h] [rbp-320h] BYREF
  _BYTE *v46; // [rsp+58h] [rbp-318h]
  __int64 v47; // [rsp+60h] [rbp-310h]
  int v48; // [rsp+68h] [rbp-308h]
  char v49; // [rsp+6Ch] [rbp-304h]
  _BYTE v50[16]; // [rsp+70h] [rbp-300h] BYREF
  unsigned __int64 v51[2]; // [rsp+80h] [rbp-2F0h] BYREF
  _BYTE v52[512]; // [rsp+90h] [rbp-2E0h] BYREF
  __int64 v53; // [rsp+290h] [rbp-E0h]
  __int64 v54; // [rsp+298h] [rbp-D8h]
  unsigned __int64 *v55; // [rsp+2A0h] [rbp-D0h]
  __int64 v56; // [rsp+2A8h] [rbp-C8h]
  char v57; // [rsp+2B0h] [rbp-C0h]
  __int64 v58; // [rsp+2B8h] [rbp-B8h]
  char *v59; // [rsp+2C0h] [rbp-B0h]
  __int64 v60; // [rsp+2C8h] [rbp-A8h]
  int v61; // [rsp+2D0h] [rbp-A0h]
  char v62; // [rsp+2D4h] [rbp-9Ch]
  char v63; // [rsp+2D8h] [rbp-98h] BYREF
  __int16 v64; // [rsp+318h] [rbp-58h]
  _QWORD *v65; // [rsp+320h] [rbp-50h]
  _QWORD *v66; // [rsp+328h] [rbp-48h]
  __int64 v67; // [rsp+330h] [rbp-40h]

  v8 = sub_BC1CD0(a4, &unk_5026090, a3);
  v12 = *(unsigned int *)(a4 + 88);
  v13 = *(_QWORD *)(a4 + 72);
  v14 = (_BYTE *)v8;
  if ( !(_DWORD)v12 )
    goto LABEL_38;
  v9 = &unk_4F81450;
  v15 = 1;
  v10 = (unsigned int)(v12 - 1);
  for ( i = v10
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v10 & v18 )
  {
    v17 = v13 + 24LL * i;
    v11 = *(void **)v17;
    if ( *(_UNKNOWN **)v17 == &unk_4F81450 && a3 == *(_QWORD *)(v17 + 8) )
      break;
    if ( v11 == (void *)-4096LL && *(_QWORD *)(v17 + 8) == -4096 )
      goto LABEL_38;
    v18 = v15 + i;
    ++v15;
  }
  if ( v17 == v13 + 24 * v12 )
  {
LABEL_38:
    v19 = 0;
  }
  else
  {
    v19 = *(unsigned __int64 **)(*(_QWORD *)(v17 + 16) + 24LL);
    if ( v19 )
      ++v19;
  }
  v55 = v19;
  v20 = (void *)(a1 + 32);
  v21 = a1 + 80;
  v51[0] = (unsigned __int64)v52;
  v51[1] = 0x1000000000LL;
  v59 = &v63;
  v53 = 0;
  v54 = 0;
  v56 = 0;
  v57 = 1;
  v58 = 0;
  v60 = 8;
  v61 = 0;
  v62 = 1;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  if ( v14[44]
    && ((*(_BYTE *)(a3 + 2) & 8) == 0
     || (v38 = v19,
         v31 = sub_B2E500(a3),
         v32 = sub_B2A630(v31),
         v19 = v38,
         v20 = (void *)(a1 + 32),
         (unsigned int)(v32 - 7) > 3)) )
  {
    if ( v19 )
      v19 = v51;
    v13 = a3;
    v37 = v20;
    if ( (unsigned __int8)sub_2FC9E30(*a2, a3, (__int64)v19, v14 + 45, v14 + 46) )
    {
      v40 = v44;
      v39 = 0;
      v41 = 2;
      v42 = 0;
      v43 = 1;
      v45 = 0;
      v46 = v50;
      v47 = 2;
      v48 = 0;
      v49 = 1;
      sub_2FC9480((__int64)&v39, (__int64)&unk_5026090, (__int64)v19, (__int64)v44, v10, (__int64)v11);
      sub_2FC9480((__int64)&v39, (__int64)&unk_4F81450, v33, v34, v35, v36);
      sub_C8CF70(a1, v37, 2, (__int64)v44, (__int64)&v39);
      v13 = a1 + 80;
      sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v50, (__int64)&v45);
      if ( !v49 )
        _libc_free((unsigned __int64)v46);
      if ( !v43 )
        _libc_free((unsigned __int64)v40);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v37;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = v21;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
      *(_QWORD *)a1 = 1;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v20;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v21;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_FFCE90((__int64)v51, v13, (__int64)v19, (__int64)v9, v10, (__int64)v11);
  sub_FFD870((__int64)v51, v13, v22, v23, v24, v25);
  sub_FFBC40((__int64)v51, v13);
  v26 = v66;
  v27 = v65;
  if ( v66 != v65 )
  {
    do
    {
      v28 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v27[7];
      *v27 = &unk_49E5048;
      if ( v28 )
        v28(v27 + 5, v27 + 5, 3);
      *v27 = &unk_49DB368;
      v29 = v27[3];
      if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
        sub_BD60C0(v27 + 1);
      v27 += 9;
    }
    while ( v26 != v27 );
    v27 = v65;
  }
  if ( v27 )
    j_j___libc_free_0((unsigned __int64)v27);
  if ( !v62 )
    _libc_free((unsigned __int64)v59);
  if ( (_BYTE *)v51[0] != v52 )
    _libc_free(v51[0]);
  return a1;
}
