// Function: sub_988010
// Address: 0x988010
//
_BOOL8 __fastcall sub_988010(__int64 a1)
{
  _BOOL8 result; // rax
  __int64 v2; // rdx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) )
        return (*(_BYTE *)(v2 + 33) & 0x20) != 0;
    }
  }
  return result;
}
