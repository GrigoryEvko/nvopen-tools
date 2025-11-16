// Function: sub_3055130
// Address: 0x3055130
//
__int64 __fastcall sub_3055130(__int64 a1, __m128i *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // r13
  bool v5; // r8
  __m128i *v6; // rbx
  unsigned __int64 v7; // rax
  char v8; // [rsp+Ch] [rbp-34h]

  result = sub_3055070(a1, (unsigned __int64 *)a2);
  if ( v3 )
  {
    v4 = (_QWORD *)v3;
    v5 = 1;
    if ( !result && v3 != a1 + 8 )
    {
      v7 = *(_QWORD *)(v3 + 32);
      v5 = a2->m128i_i64[0] < v7 || a2->m128i_i64[0] == v7 && a2->m128i_i32[2] < *(_DWORD *)(v3 + 40);
    }
    v8 = v5;
    v6 = (__m128i *)sub_22077B0(0x30u);
    v6[2] = _mm_loadu_si128(a2);
    sub_220F040(v8, (__int64)v6, v4, (_QWORD *)(a1 + 8));
    ++*(_QWORD *)(a1 + 40);
    return (__int64)v6;
  }
  return result;
}
