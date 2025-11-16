// Function: sub_AA4FF0
// Address: 0xaa4ff0
//
__int64 __fastcall sub_AA4FF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdi

  result = *(_QWORD *)(a1 + 56);
  v2 = a1 + 48;
  if ( result == v2 )
    return v2;
  while ( 1 )
  {
    if ( !result )
      BUG();
    if ( *(_BYTE *)(result - 24) != 84 )
      break;
    result = *(_QWORD *)(result + 8);
    if ( v2 == result )
      return v2;
  }
  return result;
}
