// Function: sub_C26D10
// Address: 0xc26d10
//
__int64 __fastcall sub_C26D10(_QWORD *a1, __int64 a2)
{
  __int64 result; // rax
  __m128i *v4; // rsi
  __int64 v5; // [rsp+0h] [rbp-D0h] BYREF
  char v6; // [rsp+10h] [rbp-C0h]
  __int64 v7; // [rsp+20h] [rbp-B0h] BYREF
  char v8; // [rsp+30h] [rbp-A0h]
  __int64 v9; // [rsp+40h] [rbp-90h] BYREF
  char v10; // [rsp+50h] [rbp-80h]
  _QWORD v11[4]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v12; // [rsp+80h] [rbp-50h] BYREF
  __m128i v13; // [rsp+90h] [rbp-40h] BYREF
  __int64 v14; // [rsp+A0h] [rbp-30h]

  sub_C22550((__int64)&v5, a1);
  if ( (v6 & 1) == 0 || (result = (unsigned int)v5, !(_DWORD)v5) )
  {
    v12.m128i_i32[0] = v5;
    sub_C22550((__int64)&v7, a1);
    if ( (v8 & 1) == 0 || (result = (unsigned int)v7, !(_DWORD)v7) )
    {
      v12.m128i_i64[1] = v7;
      sub_C22550((__int64)&v9, a1);
      if ( (v10 & 1) == 0 || (result = (unsigned int)v9, !(_DWORD)v9) )
      {
        v13.m128i_i64[0] = v9;
        sub_C22550((__int64)v11, a1);
        result = sub_C21E20(v11);
        if ( !(_DWORD)result )
        {
          v4 = (__m128i *)a1[51];
          v14 = a2;
          v13.m128i_i64[1] = v11[0];
          if ( v4 == (__m128i *)a1[52] )
          {
            sub_C26B60((__int64)(a1 + 50), v4, &v12);
          }
          else
          {
            if ( v4 )
            {
              *v4 = _mm_loadu_si128(&v12);
              v4[1] = _mm_loadu_si128(&v13);
              v4[2].m128i_i64[0] = v14;
              v4 = (__m128i *)a1[51];
            }
            a1[51] = (char *)v4 + 40;
          }
          sub_C1AFD0();
          return 0;
        }
      }
    }
  }
  return result;
}
