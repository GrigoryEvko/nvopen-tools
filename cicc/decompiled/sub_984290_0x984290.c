// Function: sub_984290
// Address: 0x984290
//
__int64 __fastcall sub_984290(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  *(_DWORD *)a1 |= *(_DWORD *)a2;
  result = *(unsigned __int8 *)(a1 + 5);
  if ( (_BYTE)result != *(_BYTE *)(a2 + 5) )
  {
    if ( !(_BYTE)result )
      return result;
LABEL_4:
    *(_BYTE *)(a1 + 5) = 0;
    return result;
  }
  if ( (_BYTE)result )
  {
    result = *(unsigned __int8 *)(a2 + 4);
    if ( *(_BYTE *)(a1 + 4) != (_BYTE)result )
      goto LABEL_4;
  }
  return result;
}
