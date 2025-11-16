// Function: sub_9695A0
// Address: 0x9695a0
//
__int64 __fastcall sub_9695A0(__int64 a1)
{
  __int64 result; // rax

  result = sub_C33340();
  if ( *(_QWORD *)a1 == result )
  {
    result = *(_QWORD *)(a1 + 8);
    if ( (*(_BYTE *)(result + 20) & 8) != 0 )
      return sub_C3CCB0(a1);
  }
  else if ( (*(_BYTE *)(a1 + 20) & 8) != 0 )
  {
    return sub_C34440(a1);
  }
  return result;
}
