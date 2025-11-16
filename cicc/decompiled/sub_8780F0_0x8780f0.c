// Function: sub_8780F0
// Address: 0x8780f0
//
_BOOL8 __fastcall sub_8780F0(__int64 a1)
{
  __int64 v1; // rax
  char v2; // dl
  __int64 v3; // rcx
  bool v4; // dl

  v1 = *(_QWORD *)(a1 + 88);
  if ( !v1 )
    return 0;
  do
  {
    v2 = *(_BYTE *)(v1 + 80);
    v3 = v1;
    if ( v2 == 16 )
    {
      v3 = **(_QWORD **)(v1 + 88);
      v2 = *(_BYTE *)(v3 + 80);
    }
    if ( v2 == 24 )
      v2 = *(_BYTE *)(*(_QWORD *)(v3 + 88) + 80LL);
    v1 = *(_QWORD *)(v1 + 8);
    v4 = v2 == 20;
  }
  while ( v1 && !v4 );
  return v4;
}
