// Function: sub_1441C90
// Address: 0x1441c90
//
__int64 __fastcall sub_1441C90(__int64 a1)
{
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 49) )
    return *(unsigned __int8 *)(a1 + 48);
  sub_1441BF0(a1);
  result = *(unsigned __int8 *)(a1 + 49);
  if ( (_BYTE)result )
    return *(unsigned __int8 *)(a1 + 48);
  return result;
}
