// Function: sub_7BEF80
// Address: 0x7bef80
//
_BOOL8 __fastcall sub_7BEF80(__int64 a1, __int64 a2)
{
  _BOOL8 result; // rax
  char v3; // dl

  result = 0;
  if ( *(_QWORD *)(a1 + 8) == *(_QWORD *)(a2 + 8) && *(_QWORD *)(a1 + 24) == *(_QWORD *)(a2 + 24) )
  {
    v3 = *(_BYTE *)(a2 + 33) ^ *(_BYTE *)(a1 + 33);
    if ( (v3 & 0xB) == 0 && *(_BYTE *)(a1 + 32) == *(_BYTE *)(a2 + 32) && *(_QWORD *)(a1 + 16) == *(_QWORD *)(a2 + 16) )
      return (v3 & 0x24) == 0;
  }
  return result;
}
