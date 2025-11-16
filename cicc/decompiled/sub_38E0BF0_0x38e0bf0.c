// Function: sub_38E0BF0
// Address: 0x38e0bf0
//
unsigned __int64 __fastcall sub_38E0BF0(__int64 a1, __int32 a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v5; // rbx
  __int64 (*v6)(); // rdx
  __m128i *v7; // rsi
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v9; // [rsp+10h] [rbp-30h]

  result = sub_38DD280(a1, a3);
  if ( result )
  {
    v5 = (unsigned __int64 *)result;
    v6 = *(__int64 (**)())(*(_QWORD *)a1 + 16LL);
    result = 1;
    if ( v6 != sub_38DBC10 )
      result = ((__int64 (__fastcall *)(__int64))v6)(a1);
    v8.m128i_i64[0] = result;
    v7 = (__m128i *)v5[10];
    v8.m128i_i32[2] = -1;
    v8.m128i_i32[3] = a2;
    LODWORD(v9) = 0;
    if ( v7 == (__m128i *)v5[11] )
    {
      return sub_38E08D0(v5 + 9, v7, &v8);
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128(&v8);
        result = v9;
        v7[1].m128i_i64[0] = v9;
        v7 = (__m128i *)v5[10];
      }
      v5[10] = (unsigned __int64)&v7[1].m128i_u64[1];
    }
  }
  return result;
}
