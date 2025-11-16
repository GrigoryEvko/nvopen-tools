// Function: sub_30D1130
// Address: 0x30d1130
//
__int64 __fastcall sub_30D1130(__int64 a1)
{
  __int64 result; // rax

  result = *(unsigned __int8 *)(a1 + 713);
  if ( (_BYTE)result )
    return 0;
  if ( !*(_BYTE *)(a1 + 648) && *(_DWORD *)(a1 + 716) >= *(_DWORD *)(a1 + 704) )
  {
    *(_BYTE *)(a1 + 728) = 1;
    return 1;
  }
  return result;
}
