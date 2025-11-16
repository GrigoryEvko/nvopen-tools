// Function: sub_E9BCE0
// Address: 0xe9bce0
//
__int64 __fastcall sub_E9BCE0(_QWORD *a1, unsigned int a2, _QWORD *a3)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // r13
  __int64 (*v6)(); // rax
  __m128i *v7; // rsi
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF
  __int64 v9; // [rsp+10h] [rbp-30h]

  result = sub_E99590((__int64)a1, a3);
  if ( result )
  {
    v4 = result;
    v5 = 1;
    v6 = *(__int64 (**)())(*a1 + 88LL);
    if ( v6 != sub_E97650 )
      v5 = ((__int64 (__fastcall *)(_QWORD *))v6)(a1);
    result = sub_E91EA0(*(_QWORD *)(a1[1] + 160LL), a2);
    v8.m128i_i64[0] = v5;
    v7 = *(__m128i **)(v4 + 96);
    v8.m128i_i32[2] = -1;
    v8.m128i_i32[3] = result;
    LODWORD(v9) = 0;
    if ( v7 == *(__m128i **)(v4 + 104) )
    {
      return sub_E9B9B0((const __m128i **)(v4 + 88), v7, &v8);
    }
    else
    {
      if ( v7 )
      {
        *v7 = _mm_loadu_si128(&v8);
        result = v9;
        v7[1].m128i_i64[0] = v9;
        v7 = *(__m128i **)(v4 + 96);
      }
      *(_QWORD *)(v4 + 96) = (char *)v7 + 24;
    }
  }
  return result;
}
