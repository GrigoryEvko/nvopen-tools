// Function: sub_6E1A80
// Address: 0x6e1a80
//
_BOOL8 __fastcall sub_6E1A80(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  char i; // dl

  result = 0;
  if ( !*(_BYTE *)(a1 + 8) )
  {
    v2 = *(_QWORD *)(a1 + 24);
    result = 1;
    if ( *(_BYTE *)(v2 + 24) )
    {
      v3 = *(_QWORD *)(v2 + 8);
      for ( i = *(_BYTE *)(v3 + 140); i == 12; i = *(_BYTE *)(v3 + 140) )
        v3 = *(_QWORD *)(v3 + 160);
      return i == 0;
    }
  }
  return result;
}
