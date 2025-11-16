// Function: sub_13A7B50
// Address: 0x13a7b50
//
__int64 __fastcall sub_13A7B50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_13A7AF0(a1, a2, a3);
  if ( result )
  {
    if ( *(_WORD *)(result + 24) )
      return 0;
  }
  return result;
}
