// Function: sub_74DBA0
// Address: 0x74dba0
//
__int64 __fastcall sub_74DBA0(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  __int64 v4; // r9
  __int64 v7; // r14
  const __m128i *v8; // r8
  char v9; // al
  __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // rcx
  int v13; // r14d
  __int64 v14; // rdi
  unsigned __int8 v15; // r13
  __int64 (__fastcall *v16)(__int64, __int64, _QWORD, _QWORD, _QWORD); // rax
  __int64 result; // rax
  unsigned __int8 v18; // al
  __int64 v19; // rax
  _QWORD *v20; // r11
  int v21; // eax
  int v22; // [rsp+0h] [rbp-110h]
  unsigned int v23; // [rsp+0h] [rbp-110h]
  int v24; // [rsp+0h] [rbp-110h]
  const __m128i *v25; // [rsp+8h] [rbp-108h]
  const __m128i *v26; // [rsp+8h] [rbp-108h]
  const __m128i *v27; // [rsp+8h] [rbp-108h]
  const __m128i *v28; // [rsp+8h] [rbp-108h]
  const __m128i *v29; // [rsp+8h] [rbp-108h]
  __int64 v30; // [rsp+10h] [rbp-100h]
  __int64 v31; // [rsp+10h] [rbp-100h]
  __int64 v32; // [rsp+10h] [rbp-100h]
  __int64 v33; // [rsp+10h] [rbp-100h]
  _QWORD **v34; // [rsp+10h] [rbp-100h]
  unsigned __int8 v35; // [rsp+1Fh] [rbp-F1h]
  unsigned __int8 v36; // [rsp+1Fh] [rbp-F1h]
  _OWORD v37[10]; // [rsp+20h] [rbp-F0h] BYREF
  __m128i v38; // [rsp+C0h] [rbp-50h]
  __m128i v39; // [rsp+D0h] [rbp-40h]

  v4 = a2;
  v7 = *(_QWORD *)(a1 + 128);
  v8 = (const __m128i *)v7;
  if ( *(_BYTE *)(v7 + 140) == 12 )
  {
    do
      v8 = (const __m128i *)v8[10].m128i_i64[0];
    while ( v8[8].m128i_i8[12] == 12 );
  }
  v9 = *(_BYTE *)(a1 + 192);
  v10 = *(_QWORD *)(a1 + 176);
  v11 = *(_QWORD *)(a1 + 200);
  if ( (v9 & 2) != 0 )
  {
    v35 = 11;
    if ( v11 )
    {
LABEL_5:
      if ( (*(_BYTE *)(a1 + 168) & 8) == 0 )
      {
        v13 = 0;
        if ( !a3 )
        {
          if ( !a2 )
            goto LABEL_24;
LABEL_11:
          (*(void (__fastcall **)(char *, __int64, __int64, __int64, const __m128i *))a4)("&", a4, v10, a4, v8);
          v14 = *(_QWORD *)(a1 + 184);
          v15 = v35;
          if ( !v14
            || (v16 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD, _QWORD))(a4 + 80)) == 0
            || (result = v16(v14, v11, v35, 0, 0), !(_DWORD)result) )
          {
            v18 = *(_BYTE *)(a4 + 142);
            *(_BYTE *)(a4 + 142) = 1;
            v36 = v18;
            sub_74C550(v11, v15, a4);
            result = v36;
            *(_BYTE *)(a4 + 142) = v36;
          }
          if ( a3 )
            result = (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
          goto LABEL_17;
        }
LABEL_41:
        a3 = 0;
LABEL_10:
        v24 = v4;
        v13 = a3;
        v27 = v8;
        a3 = 1;
        v32 = v10;
        (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
        v8 = v27;
        v10 = v32;
        if ( v24 )
          goto LABEL_11;
LABEL_24:
        if ( v10 && (*(_BYTE *)(v10 + 96) & 2) == 0 )
        {
          v19 = *(_QWORD *)(v10 + 112);
          v20 = *(_QWORD **)(v19 + 8);
          if ( (*(_BYTE *)(a1 + 192) & 1) != 0 )
          {
            sub_74DA20(
              *(_QWORD **)(v19 + 8),
              v8,
              (void (__fastcall **)(char *, _QWORD, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, _QWORD, __int64, __int64, __int64))a4);
          }
          else if ( *v20 )
          {
            do
            {
              v29 = v8;
              v34 = (_QWORD **)v20;
              v37[0] = _mm_loadu_si128(v8);
              v37[1] = _mm_loadu_si128(v8 + 1);
              v37[2] = _mm_loadu_si128(v8 + 2);
              v37[3] = _mm_loadu_si128(v8 + 3);
              v37[4] = _mm_loadu_si128(v8 + 4);
              v37[5] = _mm_loadu_si128(v8 + 5);
              v37[6] = _mm_loadu_si128(v8 + 6);
              v37[7] = _mm_loadu_si128(v8 + 7);
              v37[8] = _mm_loadu_si128(v8 + 8);
              v37[9] = _mm_loadu_si128(v8 + 9);
              v38 = _mm_loadu_si128(v8 + 10);
              v39 = _mm_loadu_si128(v8 + 11);
              v38.m128i_i64[0] = *(_QWORD *)(v20[2] + 40LL);
              (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
              sub_74B930((__int64)v37, a4);
              (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
              v8 = v29;
              v20 = *v34;
            }
            while ( **v34 );
          }
        }
        goto LABEL_11;
      }
      if ( !a2 )
        goto LABEL_7;
      goto LABEL_28;
    }
  }
  else
  {
    v35 = 8;
    if ( v11 )
      goto LABEL_5;
  }
  if ( (*(_BYTE *)(a1 + 168) & 8) == 0 )
    return (*(__int64 (__fastcall **)(char *, __int64))a4)("0", a4);
  v11 = 0;
  if ( !a2 )
  {
LABEL_7:
    if ( !a3 )
      goto LABEL_9;
    goto LABEL_8;
  }
LABEL_28:
  if ( (v9 & 1) != 0 )
    goto LABEL_7;
  if ( v10 )
  {
    v28 = v8;
    v33 = *(_QWORD *)(a1 + 176);
    v21 = sub_8D5E70(v10);
    v10 = v33;
    v8 = v28;
    v4 = a2;
    if ( v21 )
      goto LABEL_7;
  }
  if ( v11 )
  {
    if ( !a3 )
    {
      v13 = 0;
      goto LABEL_11;
    }
    goto LABEL_41;
  }
  if ( !a3 )
  {
    (*(void (__fastcall **)(char *, __int64, __int64, __int64, const __m128i *, __int64))a4)("(", a4, v10, a4, v8, v4);
    sub_74B930(v7, a4);
    (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
    return (*(__int64 (__fastcall **)(char *, __int64))a4)("0", a4);
  }
LABEL_8:
  v22 = v4;
  a3 = 1;
  v25 = v8;
  v30 = v10;
  (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
  LODWORD(v4) = v22;
  v8 = v25;
  v10 = v30;
LABEL_9:
  v23 = v4;
  v26 = v8;
  v31 = v10;
  (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
  sub_74B930(v7, a4);
  (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
  v10 = v31;
  v8 = v26;
  LODWORD(v4) = v23;
  if ( v11 )
    goto LABEL_10;
  v13 = a3;
  result = (*(__int64 (__fastcall **)(char *, __int64, __int64, __int64, const __m128i *, _QWORD))a4)(
             "0",
             a4,
             v31,
             v12,
             v26,
             v23);
LABEL_17:
  if ( v13 )
    return (*(__int64 (__fastcall **)(char *, __int64))a4)(")", a4);
  return result;
}
