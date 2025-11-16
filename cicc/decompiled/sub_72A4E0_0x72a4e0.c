// Function: sub_72A4E0
// Address: 0x72a4e0
//
__int64 __fastcall sub_72A4E0(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 24);
  if ( (_BYTE)result == 3 )
  {
    if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
      return sub_72A420(*(__int64 **)(a1 + 56));
  }
  else if ( (_BYTE)result == 20 )
  {
    result = *(_QWORD *)(a1 + 56);
    *(_BYTE *)(result + 192) |= 1u;
  }
  return result;
}
