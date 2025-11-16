// Function: sub_8860A0
// Address: 0x8860a0
//
__int64 __fastcall sub_8860A0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdi
  __m128i *v10; // rax
  unsigned int v11[6]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_87A120(a1, v11, a3, a4, a5, a6);
  if ( result )
  {
    v7 = *(__m128i **)(result + 48);
    if ( !v7 )
      return sub_684B10(0x599u, v11, *(_QWORD *)(result + 8));
    *(_QWORD *)(result + 48) = v7->m128i_i64[0];
    v8 = sub_87A4E0(result);
    if ( v8 )
    {
      if ( v7->m128i_i64[1] == v8 )
      {
        v10 = *(__m128i **)(v8 + 88);
        *v10 = _mm_loadu_si128(v7 + 1);
        v10[1].m128i_i64[0] = v7[2].m128i_i64[0];
LABEL_8:
        result = qword_4F5FFD0;
        v7->m128i_i64[0] = qword_4F5FFD0;
        qword_4F5FFD0 = (__int64)v7;
        return result;
      }
      sub_881DB0(v8);
    }
    v9 = v7->m128i_i64[1];
    if ( v9 )
      sub_885FF0(v9, 0, 1);
    goto LABEL_8;
  }
  return result;
}
