// Function: sub_8819D0
// Address: 0x8819d0
//
__int64 __fastcall sub_8819D0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v8; // r12
  __int64 v9; // rax
  const __m128i *v10; // roff
  unsigned int v11[6]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_87A120(a1, v11, a3, a4, a5, a6);
  if ( result )
  {
    v7 = qword_4F5FFD0;
    v8 = result;
    if ( qword_4F5FFD0 )
      qword_4F5FFD0 = *(_QWORD *)qword_4F5FFD0;
    else
      v7 = sub_823970(40);
    *(_QWORD *)v7 = 0;
    *(_QWORD *)(v7 + 8) = 0;
    sub_81B550((unsigned __int8 *)(v7 + 16));
    v9 = sub_87A4E0(v8);
    *(_QWORD *)(v7 + 8) = v9;
    if ( v9 )
    {
      v10 = *(const __m128i **)(v9 + 88);
      *(__m128i *)(v7 + 16) = _mm_loadu_si128(v10);
      *(_QWORD *)(v7 + 32) = v10[1].m128i_i64[0];
    }
    result = *(_QWORD *)(v8 + 48);
    *(_QWORD *)v7 = result;
    *(_QWORD *)(v8 + 48) = v7;
  }
  return result;
}
