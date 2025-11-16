// Function: sub_735A00
// Address: 0x735a00
//
__int64 __fastcall sub_735A00(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rbx

  while ( 1 )
  {
    result = *(unsigned __int8 *)(a2 + 24);
    if ( (unsigned __int8)result <= 6u )
      break;
    if ( (_BYTE)result != 10 )
      return result;
    a2 = *(_QWORD *)(a2 + 56);
  }
  if ( (unsigned __int8)result > 4u )
    return sub_735890(a1, *(_QWORD *)(a2 + 56), 0);
  if ( (_BYTE)result == 1 )
  {
    for ( i = *(_QWORD *)(a2 + 72); i; i = *(_QWORD *)(i + 16) )
      result = sub_735A00(a1, i);
  }
  return result;
}
