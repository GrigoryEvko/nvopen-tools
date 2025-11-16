// Function: sub_F11D70
// Address: 0xf11d70
//
__int64 __fastcall sub_F11D70(_QWORD **a1, _BYTE *a2)
{
  __int64 result; // rax

  if ( *a2 > 0x15u )
    return 0;
  **a1 = a2;
  result = 1;
  if ( *a2 <= 0x15u )
  {
    if ( *a2 == 5 )
      return 0;
    return (unsigned int)sub_AD6CA0((__int64)a2) ^ 1;
  }
  return result;
}
