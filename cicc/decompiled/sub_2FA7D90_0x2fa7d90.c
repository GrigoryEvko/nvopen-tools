// Function: sub_2FA7D90
// Address: 0x2fa7d90
//
__int64 __fastcall sub_2FA7D90(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  int v6; // eax
  int v7; // eax
  __int64 v8; // rax
  const char *v10; // rsi
  __m128i *v11; // r12
  __int64 v12; // rax
  void *v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int j; // eax
  int v18; // eax
  unsigned __int64 *v19; // r13
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  _QWORD *v25; // rbx
  _QWORD *v26; // r15
  void (__fastcall *v27)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v28; // rax
  void *v29; // [rsp+10h] [rbp-380h]
  void *v30; // [rsp+18h] [rbp-378h]
  char v31; // [rsp+27h] [rbp-369h]
  unsigned __int64 v32; // [rsp+30h] [rbp-360h]
  __int64 i; // [rsp+68h] [rbp-328h]
  __int64 v36[3]; // [rsp+70h] [rbp-320h] BYREF
  unsigned __int64 v37; // [rsp+88h] [rbp-308h]
  __int64 v38; // [rsp+90h] [rbp-300h]
  __int64 v39; // [rsp+98h] [rbp-2F8h]
  unsigned __int64 v40; // [rsp+A0h] [rbp-2F0h] BYREF
  unsigned __int64 v41; // [rsp+A8h] [rbp-2E8h]
  _DWORD v42[3]; // [rsp+B0h] [rbp-2E0h] BYREF
  char v43; // [rsp+BCh] [rbp-2D4h]
  _QWORD v44[2]; // [rsp+C0h] [rbp-2D0h] BYREF
  __int64 v45; // [rsp+D0h] [rbp-2C0h] BYREF
  _BYTE *v46; // [rsp+D8h] [rbp-2B8h]
  __int64 v47; // [rsp+E0h] [rbp-2B0h]
  int v48; // [rsp+E8h] [rbp-2A8h]
  char v49; // [rsp+ECh] [rbp-2A4h]
  _BYTE v50[448]; // [rsp+F0h] [rbp-2A0h] BYREF
  __int64 v51; // [rsp+2B0h] [rbp-E0h]
  __int64 v52; // [rsp+2B8h] [rbp-D8h]
  unsigned __int64 *v53; // [rsp+2C0h] [rbp-D0h]
  __int64 v54; // [rsp+2C8h] [rbp-C8h]
  char v55; // [rsp+2D0h] [rbp-C0h]
  __int64 v56; // [rsp+2D8h] [rbp-B8h]
  char *v57; // [rsp+2E0h] [rbp-B0h]
  __int64 v58; // [rsp+2E8h] [rbp-A8h]
  int v59; // [rsp+2F0h] [rbp-A0h]
  char v60; // [rsp+2F4h] [rbp-9Ch]
  char v61; // [rsp+2F8h] [rbp-98h] BYREF
  __int16 v62; // [rsp+338h] [rbp-58h]
  _QWORD *v63; // [rsp+340h] [rbp-50h]
  _QWORD *v64; // [rsp+348h] [rbp-48h]
  __int64 v65; // [rsp+350h] [rbp-40h]

  v4 = sub_BC0510(a4, &unk_501DA18, a3);
  v5 = *(_QWORD *)(v4 + 8) + 8LL * *(unsigned int *)(v4 + 16);
  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(v4 + 8), "shadow-stack", 0xCu, v6);
  if ( v7 == -1 )
    v8 = *(_QWORD *)(v4 + 8) + 8LL * *(unsigned int *)(v4 + 16);
  else
    v8 = *(_QWORD *)(v4 + 8) + 8LL * v7;
  v30 = (void *)(a1 + 32);
  v29 = (void *)(a1 + 80);
  if ( v5 == v8 )
  {
    memset(v36, 0, sizeof(v36));
    v37 = 0;
    v38 = 0;
    v39 = 0;
    v31 = sub_2FA7A40(v36, (__int64 **)a3);
    for ( i = *(_QWORD *)(a3 + 32); a3 + 24 != i; i = *(_QWORD *)(i + 8) )
    {
      v10 = (const char *)&unk_4F82418;
      v11 = (__m128i *)(i - 56);
      if ( !i )
        v11 = 0;
      v12 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
      v15 = *(unsigned int *)(v12 + 88);
      v16 = *(_QWORD *)(v12 + 72);
      if ( !(_DWORD)v15 )
        goto LABEL_43;
      v14 = 1;
      v32 = (unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32;
      for ( j = (v15 - 1)
              & (((0xBF58476D1CE4E5B9LL * (v32 | ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4))) >> 31)
               ^ (484763065 * (v32 | ((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)))); ; j = (v15 - 1) & v18 )
      {
        v10 = (const char *)(v16 + 24LL * j);
        v13 = *(void **)v10;
        if ( *(_UNKNOWN **)v10 == &unk_4F81450 && v11 == *((__m128i **)v10 + 1) )
          break;
        if ( v13 == (void *)-4096LL && *((_QWORD *)v10 + 1) == -4096 )
          goto LABEL_43;
        v18 = v14 + j;
        v14 = (unsigned int)(v14 + 1);
      }
      if ( v10 == (const char *)(v16 + 24 * v15) )
      {
LABEL_43:
        v19 = 0;
      }
      else
      {
        v19 = *(unsigned __int64 **)(*((_QWORD *)v10 + 2) + 24LL);
        if ( v19 )
          ++v19;
      }
      v53 = v19;
      v51 = 0;
      v40 = (unsigned __int64)v42;
      v41 = 0x1000000000LL;
      v52 = 0;
      v57 = &v61;
      if ( v19 )
        v19 = &v40;
      v54 = 0;
      v55 = 1;
      v56 = 0;
      v58 = 8;
      v59 = 0;
      v60 = 1;
      v62 = 0;
      v63 = 0;
      v64 = 0;
      v65 = 0;
      if ( (v11->m128i_i8[3] & 0x40) != 0 )
      {
        v20 = sub_B2DBE0((__int64)v11);
        v10 = "shadow-stack";
        if ( !sub_2241AC0(v20, "shadow-stack") )
        {
          v10 = (const char *)v11;
          v31 |= sub_2FA5DB0(v36, v11, (__int64)v19);
        }
      }
      sub_FFCE90((__int64)&v40, (__int64)v10, v15, v16, (__int64)v13, v14);
      sub_FFD870((__int64)&v40, (__int64)v10, v21, v22, v23, v24);
      sub_FFBC40((__int64)&v40, (__int64)v10);
      v25 = v64;
      v26 = v63;
      if ( v64 != v63 )
      {
        do
        {
          v27 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v26[7];
          *v26 = &unk_49E5048;
          if ( v27 )
            v27(v26 + 5, v26 + 5, 3);
          *v26 = &unk_49DB368;
          v28 = v26[3];
          if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
            sub_BD60C0(v26 + 1);
          v26 += 9;
        }
        while ( v25 != v26 );
        v26 = v63;
      }
      if ( v26 )
        j_j___libc_free_0((unsigned __int64)v26);
      if ( !v60 )
        _libc_free((unsigned __int64)v57);
      if ( (_DWORD *)v40 != v42 )
        _libc_free(v40);
    }
    if ( v31 )
    {
      v41 = (unsigned __int64)v44;
      v44[0] = &unk_4F81450;
      v42[0] = 2;
      v42[2] = 0;
      v43 = 1;
      v45 = 0;
      v46 = v50;
      v47 = 2;
      v48 = 0;
      v49 = 1;
      v42[1] = 1;
      v40 = 1;
      sub_C8CF70(a1, v30, 2, (__int64)v44, (__int64)&v40);
      sub_C8CF70(a1 + 48, v29, 2, (__int64)v50, (__int64)&v45);
      if ( !v49 )
        _libc_free((unsigned __int64)v46);
      if ( !v43 )
        _libc_free(v41);
    }
    else
    {
      *(_QWORD *)(a1 + 8) = v30;
      *(_QWORD *)(a1 + 48) = 0;
      *(_QWORD *)(a1 + 56) = v29;
      *(_QWORD *)(a1 + 16) = 0x100000002LL;
      *(_QWORD *)(a1 + 64) = 2;
      *(_DWORD *)(a1 + 72) = 0;
      *(_BYTE *)(a1 + 76) = 1;
      *(_DWORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 28) = 1;
      *(_QWORD *)a1 = 1;
      *(_QWORD *)(a1 + 32) = &qword_4F82400;
    }
    if ( v37 )
      j_j___libc_free_0(v37);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
