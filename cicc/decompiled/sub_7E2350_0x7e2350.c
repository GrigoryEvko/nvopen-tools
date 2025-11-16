// Function: sub_7E2350
// Address: 0x7e2350
//
__int64 __fastcall sub_7E2350(const __m128i *a1)
{
  __int64 result; // rax
  _QWORD *v2; // r13

  result = dword_4D0439C;
  if ( dword_4D0439C )
  {
    if ( a1[1].m128i_i8[8] == 1 )
    {
      result = sub_730FB0(a1[3].m128i_i8[8]);
      if ( (_DWORD)result )
      {
        result = sub_8D29A0(a1->m128i_i64[0]);
        if ( (_DWORD)result )
        {
          v2 = sub_730FF0(a1);
          *v2 = sub_72BA30(5u);
          return sub_7E2300((__int64)a1, (__int64)v2, a1->m128i_i64[0]);
        }
      }
    }
  }
  return result;
}
