// Function: sub_D34EB0
// Address: 0xd34eb0
//
__int64 __fastcall sub_D34EB0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, char a6, char a7)
{
  __int64 v9; // r15
  __int64 v10; // r8
  __m128i v12; // rax
  __int64 v13; // rcx
  __m128i v15; // [rsp+20h] [rbp-60h] BYREF
  __m128i v16; // [rsp+30h] [rbp-50h]
  __m128i v17; // [rsp+40h] [rbp-40h] BYREF

  v9 = sub_D34370(a1, a5, a3);
  if ( !(unsigned __int8)sub_DADE90(*(_QWORD *)(a1 + 112), v9, a4) )
  {
    if ( a2[8] == 18 || *(_WORD *)(v9 + 24) != 8 && (!a6 || (v9 = sub_DEF530(a1, a3)) == 0) )
    {
LABEL_10:
      v16.m128i_i8[8] = 0;
      return v16.m128i_i64[0];
    }
    if ( a4 == *(_QWORD *)(v9 + 48) )
    {
      v12.m128i_i64[0] = sub_D33F60((_QWORD *)v9, a4, (__int64)a2, *(_QWORD *)(a1 + 112), v10);
      v17 = v12;
      v13 = v12.m128i_i64[0];
      v16 = v12;
      v15 = v12;
      if ( a7 && v12.m128i_i8[8] )
      {
        if ( sub_D34050(a1, v9, a3, a2, a4, a6, v12.m128i_i64[0], v12.m128i_i8[8]) )
        {
          v16 = _mm_loadu_si128(&v15);
          return v16.m128i_i64[0];
        }
        goto LABEL_10;
      }
    }
    else
    {
      v17.m128i_i8[8] = 0;
      v12.m128i_i8[8] = 0;
      v13 = v17.m128i_i64[0];
      v15 = _mm_loadu_si128(&v17);
    }
    v15.m128i_i64[0] = v13;
    v15.m128i_i8[8] = v12.m128i_i8[8];
    v16 = _mm_loadu_si128(&v15);
    return v16.m128i_i64[0];
  }
  v16.m128i_i64[0] = 0;
  v16.m128i_i8[8] = 1;
  return v16.m128i_i64[0];
}
