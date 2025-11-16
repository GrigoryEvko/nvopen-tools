// Function: sub_8AE280
// Address: 0x8ae280
//
__int64 __fastcall sub_8AE280(
        __int64 a1,
        __m128i **a2,
        int *a3,
        _DWORD *a4,
        _DWORD *a5,
        int *a6,
        _DWORD *a7,
        int a8,
        _QWORD *a9)
{
  __int16 v12; // r9
  __int64 v13; // rdi
  __int64 v14; // rax
  int v15; // edx
  unsigned __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BOOL8 v21; // rsi
  const __m128i *v22; // rdi
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdi
  char v27; // dl
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // r14
  const __m128i *v31; // r13
  int v32; // esi
  _QWORD v36[66]; // [rsp+20h] [rbp-210h] BYREF

  memset(v36, 0, 0x1D8u);
  v36[19] = v36;
  v36[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v36[22]) |= 1u;
  BYTE4(v36[16]) |= 0x84u;
  v36[15] = v36[15] & 0xFFFFFFBFD7FFFFFFLL
          | ((unsigned __int64)(dword_4D04804 & 1) << 38)
          | ((unsigned __int64)(word_4D04430 & 1) << 27)
          | ((unsigned __int64)(dword_4D04408 & 1) << 29);
  sub_672A20(0x102u, (__int64)v36, (__int64)a9, (__int64)&dword_4D04804, (unsigned __int64)a5);
  v12 = v36[1];
  if ( (v36[1] & 0x20) != 0 )
  {
    sub_6851C0(0xFFu, &v36[3]);
    v29 = sub_72C930();
    v12 = v36[1];
    v36[34] = v29;
    v13 = v29;
    v36[35] = v29;
    v36[36] = v29;
  }
  else
  {
    v13 = v36[36];
  }
  if ( (v12 & 1) == 0 )
  {
    sub_64E990((__int64)dword_4F07508, v13, 0, 0, 0, (HIBYTE(v12) ^ 1) & 1);
    v13 = v36[36];
  }
  v14 = sub_8D4940(v13);
  v15 = 0;
  v16 = 131075;
  if ( *(_BYTE *)(v14 + 140) == 14 && (*(_WORD *)(v14 + 160) & 0x1FF) == 0x100 )
  {
    v28 = *(_DWORD *)(*(_QWORD *)(v14 + 168) + 28LL);
    if ( a8 == v28 || v28 <= 0 )
    {
      v15 = 0;
      v16 = 131075;
    }
    else
    {
      v15 = 1;
      v16 = 2228227;
    }
  }
  if ( a6 )
    *a6 = v15;
  v17 = (__int64)v36;
  sub_626F50(v16, (__int64)v36, 0, a1, 0, a9);
  if ( a5 )
    *a5 = (v36[15] & 0x40000000LL) != 0;
  if ( a3 )
    *a3 = ((unsigned __int8)(v36[2] >> 1) ^ 1) & 1;
  if ( v36[44] )
  {
    v17 = (unsigned int)dword_4F04C64;
    sub_869FD0((_QWORD *)v36[44], dword_4F04C64);
    v36[44] = 0;
  }
  sub_65C470((__int64)v36, v17, v18, v19, v20);
  if ( (v36[15] & 0x8010000000LL) == 0x8000000000LL && dword_4D04804 )
  {
    v26 = v36[36];
    if ( !a7 )
    {
      v24 = sub_8D4940(v36[36]);
      if ( *(_BYTE *)(v24 + 140) != 14 )
      {
        if ( !dword_4F077BC )
          goto LABEL_35;
        goto LABEL_31;
      }
      v25 = *(_QWORD *)(v24 + 168);
      if ( *(_DWORD *)(v25 + 28) != -1 )
      {
LABEL_30:
        if ( !dword_4F077BC )
          goto LABEL_23;
LABEL_31:
        if ( (_DWORD)qword_4F077B4 || !qword_4F077A8 )
          goto LABEL_23;
        goto LABEL_48;
      }
      goto LABEL_41;
    }
    *a7 = 1;
    v24 = sub_8D4940(v26);
    if ( *(_BYTE *)(v24 + 140) == 14 )
    {
      v25 = *(_QWORD *)(v24 + 168);
      if ( *(_DWORD *)(v25 + 28) == -1 )
      {
LABEL_41:
        v27 = *(_BYTE *)(v24 + 161) | 4;
        *(_BYTE *)(v24 + 161) = v27;
        *(_BYTE *)(v24 + 161) = (8 * (*(_DWORD *)(v25 + 24) == 2)) | v27 & 0xF7;
      }
    }
  }
  if ( !a4 || !a7 )
    goto LABEL_30;
  *a4 = sub_8DC060(v36[36]);
  if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
    goto LABEL_24;
  if ( !qword_4F077A8 )
    goto LABEL_23;
LABEL_48:
  v30 = v36[36];
  if ( *(_BYTE *)(v36[36] + 140LL) == 12 && (*(_BYTE *)(v36[36] + 186LL) & 0x20) != 0 )
  {
    v31 = (const __m128i *)v36[36];
    do
      v31 = (const __m128i *)v31[10].m128i_i64[0];
    while ( v31[8].m128i_i8[12] == 12 );
    if ( !(unsigned int)sub_8DBE70(v31) )
    {
      if ( (*(_BYTE *)(v30 + 140) & 0xFB) == 8 )
        v32 = sub_8D4C10(v30, dword_4F077C4 != 2);
      else
        v32 = 0;
      v36[36] = sub_73C570(v31, v32);
    }
  }
LABEL_23:
  if ( a7 )
  {
LABEL_24:
    v21 = *a7 != 0;
    goto LABEL_25;
  }
LABEL_35:
  v21 = 0;
LABEL_25:
  if ( (unsigned int)sub_8AE140(&v36[36], v21, &v36[3]) )
  {
    v22 = (const __m128i *)v36[36];
  }
  else
  {
    v36[34] = sub_72C930();
    v22 = (const __m128i *)v36[34];
    v36[35] = v36[34];
    v36[36] = v36[34];
  }
  *a2 = sub_73D4C0(v22, dword_4F077C4 == 2);
  sub_644920(v36, 1);
  sub_643EB0((__int64)v36, 0);
  return sub_7604D0(v36[36], 6u);
}
