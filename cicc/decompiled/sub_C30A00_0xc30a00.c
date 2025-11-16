// Function: sub_C30A00
// Address: 0xc30a00
//
__int64 __fastcall sub_C30A00(__int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int32 v4; // r15d
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __int64 v10; // rax
  __int64 result; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __m128i v15; // xmm3
  __int64 v16; // rax
  __int32 v17; // [rsp+Ch] [rbp-D4h]
  char v18; // [rsp+1Fh] [rbp-C1h] BYREF
  __m128i v19; // [rsp+20h] [rbp-C0h] BYREF
  __m128i v20; // [rsp+30h] [rbp-B0h] BYREF
  __m128i v21; // [rsp+40h] [rbp-A0h] BYREF
  __m128i v22; // [rsp+50h] [rbp-90h] BYREF
  __int64 v23; // [rsp+60h] [rbp-80h] BYREF
  __int64 v24; // [rsp+68h] [rbp-78h]
  __m128i v25; // [rsp+70h] [rbp-70h] BYREF
  __m128i v26; // [rsp+80h] [rbp-60h]
  __m128i v27; // [rsp+90h] [rbp-50h] BYREF
  __int64 v28; // [rsp+A8h] [rbp-38h]

  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!Passed",
          7,
          *(_DWORD *)*a2 == 1)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!Missed",
          7,
          *(_DWORD *)*a2 == 2)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!Analysis",
          9,
          *(_DWORD *)*a2 == 3)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!AnalysisFPCommute",
          18,
          *(_DWORD *)*a2 == 4)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!AnalysisAliasing",
          17,
          *(_DWORD *)*a2 == 5)
    && !(*(unsigned __int8 (__fastcall **)(__int64 *, const char *, __int64, bool))(*a1 + 96))(
          a1,
          "!Failure",
          8,
          *(_DWORD *)*a2 == 6) )
  {
    BUG();
  }
  v2 = sub_CB0A70(a1);
  if ( *(_DWORD *)(v2 + 8) == 2 )
  {
    v3 = v2 + 32;
    sub_F02AE0(&v27, v2 + 32, *(_QWORD *)(*a2 + 8), *(_QWORD *)(*a2 + 16));
    v17 = v27.m128i_i32[0];
    sub_F02AE0(&v27, v3, *(_QWORD *)(*a2 + 24), *(_QWORD *)(*a2 + 32));
    v4 = v27.m128i_i32[0];
    sub_F02AE0(&v27, v3, *(_QWORD *)(*a2 + 40), *(_QWORD *)(*a2 + 48));
    v5 = *a2;
    v6 = *(_QWORD *)(*a2 + 104);
    v7 = *(unsigned int *)(*a2 + 112);
    v20.m128i_i32[0] = v4;
    v19.m128i_i32[0] = v17;
    v25 = _mm_loadu_si128((const __m128i *)(v5 + 56));
    v8 = _mm_loadu_si128((const __m128i *)(v5 + 72));
    v21.m128i_i32[0] = v27.m128i_i32[0];
    v26 = v8;
    v9 = _mm_loadu_si128((const __m128i *)(v5 + 88));
    v10 = *a1;
    v23 = v6;
    v24 = v7;
    v22 = v9;
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(v10 + 120))(
           a1,
           "Pass",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C2F5F0(a1, (__int64)&v19);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(*a1 + 120))(
           a1,
           "Name",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C2F5F0(a1, (__int64)&v20);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
    v28 = 0;
    sub_C302C0(a1, (__int64)"DebugLoc", &v25, &v27, 0);
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(*a1 + 120))(
           a1,
           "Function",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C2F5F0(a1, (__int64)&v21);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
  }
  else
  {
    v12 = *a2;
    v13 = *(_QWORD *)(*a2 + 104);
    v14 = *(unsigned int *)(*a2 + 112);
    v19 = _mm_loadu_si128((const __m128i *)(*a2 + 8));
    v20 = _mm_loadu_si128((const __m128i *)(v12 + 24));
    v25 = _mm_loadu_si128((const __m128i *)(v12 + 56));
    v26 = _mm_loadu_si128((const __m128i *)(v12 + 72));
    v21 = _mm_loadu_si128((const __m128i *)(v12 + 40));
    v15 = _mm_loadu_si128((const __m128i *)(v12 + 88));
    v16 = *a1;
    v23 = v13;
    v24 = v14;
    v22 = v15;
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(v16 + 120))(
           a1,
           "Pass",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C300B0(a1, (__int64)&v19);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(*a1 + 120))(
           a1,
           "Name",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C300B0(a1, (__int64)&v20);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
    v28 = 0;
    sub_C302C0(a1, (__int64)"DebugLoc", &v25, &v27, 0);
    if ( (*(unsigned __int8 (__fastcall **)(__int64 *, char *, __int64, _QWORD, char *, __m128i *))(*a1 + 120))(
           a1,
           "Function",
           1,
           0,
           &v18,
           &v27) )
    {
      sub_C300B0(a1, (__int64)&v21);
      (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
  }
  v27.m128i_i64[1] = 0;
  sub_C2F9B0(a1, (__int64)"Hotness", (__int64)&v22, &v27, 0);
  if ( !(*(unsigned __int8 (__fastcall **)(__int64 *))(*a1 + 56))(a1) || (result = v24 << 6) != 0 )
  {
    result = (*(__int64 (__fastcall **)(__int64 *, char *, _QWORD, _QWORD, char *, __m128i *))(*a1 + 120))(
               a1,
               "Args",
               0,
               0,
               &v18,
               &v27);
    if ( (_BYTE)result )
    {
      sub_C30940(a1, (__int64)&v23);
      return (*(__int64 (__fastcall **)(__int64 *, __int64))(*a1 + 128))(a1, v27.m128i_i64[0]);
    }
  }
  return result;
}
