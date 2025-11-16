// Function: sub_35EC370
// Address: 0x35ec370
//
__int64 __fastcall sub_35EC370(__int64 a1, unsigned int a2, int a3)
{
  __int64 result; // rax

  if ( *(_DWORD *)(a1 + 6432) == a2 )
  {
    *(_DWORD *)(a1 + 6440) = a3;
    *(_DWORD *)(a1 + 6444) = a2;
    *(_DWORD *)(a1 + 6448) = a3;
  }
  else if ( *(_DWORD *)(a1 + 6440) > (unsigned int)a3 )
  {
    result = (unsigned int)(a3 + dword_5040448);
    if ( (unsigned int)result <= *(_DWORD *)(a1 + 6448) )
      return sub_35EBE30(a1, a2, a3);
  }
  return result;
}
