// Function: sub_13A8380
// Address: 0x13a8380
//
__int64 __fastcall sub_13A8380(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v6; // r8
  __int64 result; // rax

  v6 = sub_13A7760(a1, 32, a2, a3);
  result = 0;
  if ( !v6 )
  {
    result = sub_13A7760(a1, 33, a2, a3);
    if ( !(_BYTE)result )
      *(_BYTE *)(a4 + 43) = 0;
  }
  return result;
}
