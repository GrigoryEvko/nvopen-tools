// Function: sub_3701890
// Address: 0x3701890
//
__int64 __fastcall sub_3701890(_QWORD *a1, unsigned __int64 *a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v9; // rdi
  unsigned __int64 v10; // rsi
  __int64 v11; // rax
  void (__fastcall *v12)(__int64 *, __int64, __int64); // rcx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdi
  __int64 result; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __m128i v30; // [rsp+0h] [rbp-50h] BYREF
  __m128i v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-30h]

  v9 = (__int64 *)a1[7];
  v10 = *a2;
  v11 = *v9;
  if ( v10 <= 0x7FFF )
  {
    if ( !a1[5] && !a1[6] )
    {
      if ( (*(unsigned __int8 (**)(void))(v11 + 40))() )
      {
        v27 = a3[2].m128i_i64[0];
        v30 = _mm_loadu_si128(a3);
        v32 = v27;
        v31 = _mm_loadu_si128(a3 + 1);
        if ( (unsigned __int8)v27 > 1u )
          (*(void (__fastcall **)(_QWORD, __m128i *))(*(_QWORD *)a1[7] + 24LL))(a1[7], &v30);
      }
      v9 = (__int64 *)a1[7];
      v10 = *a2;
    }
    result = (*(__int64 (__fastcall **)(__int64 *, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*v9 + 8))(
               v9,
               v10,
               2,
               a4,
               a5,
               a6,
               v30.m128i_i64[0],
               v30.m128i_i64[1],
               v31.m128i_i64[0],
               v31.m128i_i64[1],
               v32);
    if ( a1[7] && !a1[5] && !a1[6] )
      a1[8] += 2LL;
  }
  else
  {
    v12 = *(void (__fastcall **)(__int64 *, __int64, __int64))(v11 + 8);
    if ( v10 + 128 <= 0xFF )
    {
      v12(v9, 0x8000, 2);
      v16 = a1[7];
      if ( v16 && !a1[5] && !a1[6] )
      {
        if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v16 + 40LL))(v16) )
        {
          v26 = a3[2].m128i_i64[0];
          v30 = _mm_loadu_si128(a3);
          v32 = v26;
          v31 = _mm_loadu_si128(a3 + 1);
          if ( (unsigned __int8)v26 > 1u )
            (*(void (__fastcall **)(_QWORD, __m128i *))(*(_QWORD *)a1[7] + 24LL))(a1[7], &v30);
        }
        v16 = a1[7];
      }
      result = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v16 + 8LL))(
                 v16,
                 *a2,
                 1,
                 v13,
                 v14,
                 v15,
                 v30.m128i_i64[0],
                 v30.m128i_i64[1],
                 v31.m128i_i64[0],
                 v31.m128i_i64[1],
                 v32);
      if ( a1[7] && !a1[5] && !a1[6] )
        a1[8] += 3LL;
      return result;
    }
    if ( v10 + 0x8000 > 0xFFFF )
    {
      if ( v10 + 0x80000000 > 0xFFFFFFFF )
      {
        v12(v9, 32777, 2);
        v21 = a1[7];
        if ( v21 )
          goto LABEL_11;
      }
      else
      {
        v12(v9, 32771, 2);
        v21 = a1[7];
        if ( v21 )
        {
LABEL_11:
          if ( !a1[5] && !a1[6] )
          {
            if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 40LL))(v21) )
            {
              v29 = a3[2].m128i_i64[0];
              v30 = _mm_loadu_si128(a3);
              v32 = v29;
              v31 = _mm_loadu_si128(a3 + 1);
              if ( (unsigned __int8)v29 > 1u )
                (*(void (__fastcall **)(_QWORD, __m128i *))(*(_QWORD *)a1[7] + 24LL))(a1[7], &v30);
            }
            v21 = a1[7];
          }
        }
      }
      result = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v21 + 8LL))(
                 v21,
                 *a2,
                 4,
                 v18,
                 v19,
                 v20,
                 v30.m128i_i64[0],
                 v30.m128i_i64[1],
                 v31.m128i_i64[0],
                 v31.m128i_i64[1],
                 v32);
      if ( a1[7] && !a1[5] && !a1[6] )
        a1[8] += 6LL;
      return result;
    }
    v12(v9, 32769, 2);
    v25 = a1[7];
    if ( v25 && !a1[5] && !a1[6] )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v25 + 40LL))(v25) )
      {
        v28 = a3[2].m128i_i64[0];
        v30 = _mm_loadu_si128(a3);
        v32 = v28;
        v31 = _mm_loadu_si128(a3 + 1);
        if ( (unsigned __int8)v28 > 1u )
          (*(void (__fastcall **)(_QWORD, __m128i *))(*(_QWORD *)a1[7] + 24LL))(a1[7], &v30);
      }
      v25 = a1[7];
    }
    result = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v25 + 8LL))(
               v25,
               *a2,
               2,
               v22,
               v23,
               v24,
               v30.m128i_i64[0],
               v30.m128i_i64[1],
               v31.m128i_i64[0],
               v31.m128i_i64[1],
               v32);
    if ( a1[7] && !a1[5] && !a1[6] )
      a1[8] += 4LL;
  }
  return result;
}
