// Function: sub_C24760
// Address: 0xc24760
//
__int64 __fastcall sub_C24760(const __m128i **a1)
{
  const __m128i *v1; // rax
  __int64 result; // rax
  unsigned int v3; // ebx

  v1 = a1[9];
  a1[26] = (const __m128i *)v1->m128i_i64[1];
  a1[27] = (const __m128i *)v1[1].m128i_i64[0];
  result = sub_C22170(a1);
  if ( !(_DWORD)result )
  {
    v3 = sub_C24440(a1);
    if ( v3 )
      return v3;
    v3 = sub_C22F50((__int64)a1);
    if ( v3 )
    {
      return v3;
    }
    else
    {
      sub_C1AFD0();
      return 0;
    }
  }
  return result;
}
