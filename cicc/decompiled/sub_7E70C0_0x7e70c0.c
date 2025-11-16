// Function: sub_7E70C0
// Address: 0x7e70c0
//
__int64 __fastcall sub_7E70C0(const __m128i **a1, _DWORD *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 i; // rbx

  *a2 = 0;
  result = (__int64)qword_4D03F68;
  for ( i = qword_4D03F68[5]; i; i = *(_QWORD *)(i + 32) )
  {
    while ( 1 )
    {
      result = *(_DWORD *)(i + 48) & 0x30035000;
      if ( (_DWORD)result == 4096 )
        break;
      i = *(_QWORD *)(i + 32);
      if ( !i )
        return result;
    }
    if ( !*a2 )
    {
      *a2 = 1;
      sub_7E7090(*a1, a3, (__m128i **)a1);
    }
    result = sub_7FE8B0(i, a3);
  }
  return result;
}
