// Function: sub_6EB660
// Address: 0x6eb660
//
_BOOL8 __fastcall sub_6EB660(__int64 a1)
{
  __int64 v1; // r8
  _BOOL8 result; // rax

  v1 = sub_6EB5C0(a1);
  result = 1;
  if ( !v1 )
  {
    result = 0;
    if ( *(_BYTE *)(a1 + 16) == 2 && *(_BYTE *)(a1 + 317) == 7 && (*(_BYTE *)(a1 + 336) & 2) != 0 )
      return *(_QWORD *)(a1 + 344) != 0;
  }
  return result;
}
