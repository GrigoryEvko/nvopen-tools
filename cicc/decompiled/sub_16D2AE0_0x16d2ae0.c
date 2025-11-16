// Function: sub_16D2AE0
// Address: 0x16d2ae0
//
__int64 __fastcall sub_16D2AE0(__m128i *a1, unsigned int a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // [rsp+8h] [rbp-28h] BYREF
  __m128i v8; // [rsp+10h] [rbp-20h] BYREF

  v4 = a1->m128i_i64[1];
  if ( v4 && *(_BYTE *)a1->m128i_i64[0] == 45 )
  {
    v8.m128i_i64[0] = a1->m128i_i64[0] + 1;
    v8.m128i_i64[1] = v4 - 1;
    result = sub_16D2A00((__int64)&v8, a2, (unsigned __int64 *)&v7);
    if ( !(_BYTE)result )
    {
      v6 = -v7;
      if ( -v7 < 0 || v7 == 0 )
      {
        *a1 = _mm_loadu_si128(&v8);
LABEL_5:
        *a3 = v6;
        return result;
      }
    }
  }
  else
  {
    result = sub_16D2A00((__int64)a1, a2, (unsigned __int64 *)&v7);
    if ( !(_BYTE)result )
    {
      v6 = v7;
      if ( v7 >= 0 )
        goto LABEL_5;
    }
  }
  return 1;
}
