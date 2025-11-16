// Function: sub_74EE00
// Address: 0x74ee00
//
_QWORD *__fastcall sub_74EE00(const __m128i *a1, int a2, int a3, __int64 a4)
{
  const __m128i *v4; // r13
  __m128i *v6; // rax
  __int64 v7; // r8
  __int64 v8; // r15
  char i; // bl
  __int64 v10; // rcx
  int v11; // r14d
  char *v12; // rdi
  void (__fastcall *v13)(char *, __int64); // rax
  void (__fastcall *v14)(char *, __int64); // rax
  __int64 v16; // r14
  void (__fastcall *v17)(char *, __int64); // r14
  char *v18; // rax
  __int64 v19; // rax
  void (__fastcall *v20)(char *, __int64); // rax
  void (__fastcall *v21)(char *, __int64); // rax
  void (__fastcall *v22)(char *, __int64); // rax
  __m128i *v23; // rax
  __m128i *v24; // rax
  void (__fastcall *v25)(char *, __int64); // r15
  char *v26; // rax
  __int64 v27; // r15
  int v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+8h] [rbp-88h]
  unsigned __int8 v30; // [rsp+17h] [rbp-79h]
  int v32; // [rsp+1Ch] [rbp-74h]
  int v33; // [rsp+24h] [rbp-6Ch] BYREF
  __m128i *v34; // [rsp+28h] [rbp-68h] BYREF
  __m128i v35; // [rsp+30h] [rbp-60h] BYREF
  __int16 v36[8]; // [rsp+40h] [rbp-50h] BYREF
  _WORD v37[32]; // [rsp+50h] [rbp-40h] BYREF

  v4 = a1;
  v6 = (__m128i *)sub_724DC0();
  v8 = a1[8].m128i_i64[0];
  v34 = v6;
  for ( i = *(_BYTE *)(v8 + 140); i == 12; i = *(_BYTE *)(v8 + 140) )
    v8 = *(_QWORD *)(v8 + 160);
  if ( i == 2 )
  {
    v30 = *(_BYTE *)(v8 + 160);
    v10 = v30;
    v32 = byte_4B6DF90[v30];
    if ( (unsigned __int8)(v30 - 11) <= 1u && *(_BYTE *)(a4 + 136) )
    {
      (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
      v16 = a1[8].m128i_i64[0];
      (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
      sub_74B930(v16, a4);
      (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
      sub_620D80(v36, 0);
      sub_620DC0((__int64)&v35, (__int64)a1[11].m128i_i64);
      sub_6214E0(v35.m128i_i16, 64, v32, 1);
      if ( (unsigned int)sub_621000(v35.m128i_i16, v32, v36, v32) )
      {
        v25 = *(void (__fastcall **)(char *, __int64))a4;
        v26 = sub_622500(&v35, v32);
        v25(v26, a4);
        (*(void (__fastcall **)(const char *, __int64))a4)("<<64 | ", a4);
        v27 = a1[8].m128i_i64[0];
        (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
        sub_74B930(v27, a4);
        (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
      }
      sub_620DC0((__int64)&v35, (__int64)a1[11].m128i_i64);
      sub_620D80(v37, 0);
      sub_621DB0(v37);
      sub_621410((__int64)v37, 64, &v33);
      sub_621DB0(v37);
      sub_6213D0((__int64)&v35, (__int64)v37);
      v17 = *(void (__fastcall **)(char *, __int64))a4;
      v18 = sub_622500(&v35, v32);
      v17(v18, a4);
      goto LABEL_31;
    }
    if ( a2 )
      goto LABEL_6;
    if ( (*(_BYTE *)(v8 + 161) & 8) != 0 && (!*(_BYTE *)(a4 + 136) || !*(_BYTE *)(a4 + 141) || !*(_BYTE *)(a4 + 137))
      || v30 <= 4u && !*(_BYTE *)(a4 + 147) )
    {
LABEL_62:
      if ( a3 )
      {
        v11 = 1;
        (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
      }
      else
      {
        a3 = 1;
        v11 = 0;
      }
      v29 = a1[8].m128i_i64[0];
      (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
      sub_74B930(v29, a4);
      (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
      goto LABEL_7;
    }
LABEL_33:
    if ( v32 )
    {
      v11 = 0;
      v28 = v32;
      if ( a1[10].m128i_i8[13] != 8 )
        goto LABEL_43;
    }
    else
    {
      if ( *(_BYTE *)(a4 + 137) )
        goto LABEL_62;
      if ( a1[10].m128i_i8[13] != 8 )
      {
        v11 = 0;
LABEL_9:
        v28 = 0;
        a3 = 0;
        goto LABEL_10;
      }
    }
    sub_749FD0((__int64)a1, a3, (void (__fastcall **)(char *, _QWORD))a4, v10, v7);
    return sub_724E30((__int64)&v34);
  }
  v30 = 13;
  v32 = sub_6210B0((__int64)a1, 0) == 0;
  if ( !a2 )
    goto LABEL_33;
LABEL_6:
  v11 = 0;
LABEL_7:
  if ( a1[10].m128i_i8[13] == 8 )
  {
    sub_749FD0((__int64)a1, a3, (void (__fastcall **)(char *, _QWORD))a4, v10, v7);
    goto LABEL_25;
  }
  if ( !v32 )
    goto LABEL_9;
  v28 = v32;
LABEL_43:
  if ( (int)sub_6210B0((__int64)a1, 0) < 0 )
  {
    if ( a3 )
    {
      (*(void (__fastcall **)(char *, __int64))a4)("(", a4);
      a3 = 1;
    }
    v32 = 0;
    if ( !*(_BYTE *)(a4 + 136) )
    {
LABEL_40:
      v12 = sub_6228C0(v4);
      v13 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
      if ( v13 )
        goto LABEL_12;
      goto LABEL_41;
    }
    v23 = v34;
    *v34 = _mm_loadu_si128(a1);
    v23[1] = _mm_loadu_si128(a1 + 1);
    v23[2] = _mm_loadu_si128(a1 + 2);
    v23[3] = _mm_loadu_si128(a1 + 3);
    v23[4] = _mm_loadu_si128(a1 + 4);
    v23[5] = _mm_loadu_si128(a1 + 5);
    v23[6] = _mm_loadu_si128(a1 + 6);
    v23[7] = _mm_loadu_si128(a1 + 7);
    v23[8] = _mm_loadu_si128(a1 + 8);
    v23[9] = _mm_loadu_si128(a1 + 9);
    v23[10] = _mm_loadu_si128(a1 + 10);
    v23[11] = _mm_loadu_si128(a1 + 11);
    v23[12] = _mm_loadu_si128(a1 + 12);
    sub_621710(v34[11].m128i_i16, (_BOOL4 *)&v33);
    v32 = v33;
    if ( v33 || !sub_6211E0(v34[11].m128i_i16, 1, v30) )
    {
      v24 = v34;
      *v34 = _mm_loadu_si128(a1);
      v24[1] = _mm_loadu_si128(a1 + 1);
      v24[2] = _mm_loadu_si128(a1 + 2);
      v24[3] = _mm_loadu_si128(a1 + 3);
      v24[4] = _mm_loadu_si128(a1 + 4);
      v24[5] = _mm_loadu_si128(a1 + 5);
      v24[6] = _mm_loadu_si128(a1 + 6);
      v24[7] = _mm_loadu_si128(a1 + 7);
      v24[8] = _mm_loadu_si128(a1 + 8);
      v24[9] = _mm_loadu_si128(a1 + 9);
      v24[10] = _mm_loadu_si128(a1 + 10);
      v24[11] = _mm_loadu_si128(a1 + 11);
      v4 = v34;
      v24[12] = _mm_loadu_si128(a1 + 12);
      sub_621300((unsigned __int16 *)&v4[11]);
      v32 = 1;
    }
  }
  else
  {
    a3 = 0;
    v32 = 0;
  }
LABEL_10:
  if ( !*(_BYTE *)(a4 + 136) )
    goto LABEL_40;
  v12 = sub_622850(v4);
  v13 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
  if ( v13 )
  {
LABEL_12:
    v13(v12, a4);
    goto LABEL_13;
  }
LABEL_41:
  (*(void (__fastcall **)(char *, __int64))a4)(v12, a4);
LABEL_13:
  if ( !*(_BYTE *)(a4 + 157) )
  {
    if ( !v28 && !*(_BYTE *)(a4 + 137) )
    {
      v14 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
      if ( !v14 )
        v14 = *(void (__fastcall **)(char *, __int64))a4;
      v14("U", a4);
    }
    if ( i == 2 )
    {
      if ( (unsigned __int8)(v30 - 7) > 1u )
      {
        if ( (unsigned __int8)(v30 - 9) > 1u )
          goto LABEL_21;
        goto LABEL_47;
      }
    }
    else
    {
      if ( i != 6 )
        goto LABEL_21;
      v19 = *(_QWORD *)(v8 + 128);
      if ( v19 != 4 )
      {
        if ( v19 != 8 )
          goto LABEL_21;
LABEL_47:
        if ( *(_BYTE *)(a4 + 136) && unk_4F068C0 && !sub_5D76E0() )
        {
          v20 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
          if ( !v20 )
            v20 = *(void (__fastcall **)(char *, __int64))a4;
          v20("i64", a4);
        }
        else
        {
          v21 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
          if ( !v21 )
            v21 = *(void (__fastcall **)(char *, __int64))a4;
          v21("LL", a4);
        }
        goto LABEL_21;
      }
    }
    v22 = *(void (__fastcall **)(char *, __int64))(a4 + 8);
    if ( !v22 )
      v22 = *(void (__fastcall **)(char *, __int64))a4;
    v22("L", a4);
  }
LABEL_21:
  if ( v32 )
    (*(void (__fastcall **)(const char *, __int64))a4)("-1", a4);
  if ( a3 )
    (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
LABEL_25:
  if ( v11 )
LABEL_31:
    (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
  return sub_724E30((__int64)&v34);
}
