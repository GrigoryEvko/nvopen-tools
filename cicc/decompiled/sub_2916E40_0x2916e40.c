// Function: sub_2916E40
// Address: 0x2916e40
//
int __fastcall sub_2916E40(__int64 a1, unsigned __int8 *a2, const __m128i *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  char v7; // cl
  __int64 v9; // rdi
  __m128i v10; // [rsp+0h] [rbp-40h] BYREF
  __m128i v11; // [rsp+10h] [rbp-30h]
  __int64 v12; // [rsp+20h] [rbp-20h]

  v7 = a3[2].m128i_i8[0];
  if ( (unsigned __int8)v7 > 1u )
  {
    v9 = a1 + 8;
    if ( a3[2].m128i_i8[1] == 1 )
    {
      v5 = a3->m128i_i64[1];
      a3 = (const __m128i *)a3->m128i_i64[0];
    }
    else
    {
      v7 = 2;
    }
    v10.m128i_i64[0] = v9;
    v11.m128i_i64[0] = (__int64)a3;
    v11.m128i_i64[1] = v5;
    LOBYTE(v12) = 4;
    BYTE1(v12) = v7;
  }
  else
  {
    v12 = a3[2].m128i_i64[0];
    v10 = _mm_loadu_si128(a3);
    v11 = _mm_loadu_si128(a3 + 1);
  }
  if ( a4 )
    sub_B44240(a2, *(_QWORD *)(a4 + 16), (unsigned __int64 *)a4, a5);
  return sub_BD6B50(a2, (const char **)&v10);
}
