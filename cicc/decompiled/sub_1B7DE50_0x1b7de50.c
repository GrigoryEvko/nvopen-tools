// Function: sub_1B7DE50
// Address: 0x1b7de50
//
__int64 __fastcall sub_1B7DE50(__int64 a1, unsigned int a2)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 54 )
  {
    sub_15F8F50(a1, a2);
    result = *(unsigned __int8 *)(a1 + 16);
  }
  if ( (_BYTE)result == 55 )
  {
    sub_15F9450(a1, a2);
    result = *(unsigned __int8 *)(a1 + 16);
  }
  if ( (_BYTE)result == 78 )
    return sub_1B7DC70(a1, a2);
  return result;
}
