// Function: sub_860080
// Address: 0x860080
//
_BOOL8 __fastcall sub_860080(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  char v3; // dl
  _BOOL4 v4; // r8d

  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
    v3 = *(_BYTE *)(*(_QWORD *)(v2 + 168) + 109LL) & 0x20;
    v4 = v3 != 0;
    if ( !v3 )
      v2 = 0;
  }
  else
  {
    v4 = 0;
    v2 = 0;
  }
  if ( a2 )
    *a2 = v2;
  return v4;
}
