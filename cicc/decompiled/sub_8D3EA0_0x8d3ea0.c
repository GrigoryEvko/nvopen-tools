// Function: sub_8D3EA0
// Address: 0x8d3ea0
//
_BOOL8 __fastcall sub_8D3EA0(__int64 a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( *(_BYTE *)(a1 + 140) == 14 && !*(_BYTE *)(a1 + 160) )
    return *(_DWORD *)(*(_QWORD *)(a1 + 168) + 28LL) == -1;
  return result;
}
