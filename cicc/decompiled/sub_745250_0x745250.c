// Function: sub_745250
// Address: 0x745250
//
__int64 __fastcall sub_745250(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // rbx

  v2 = *(_QWORD *)(a1 + 128);
  for ( *a2 = 0; *(_BYTE *)(v2 + 140) == 12; v2 = *(_QWORD *)(v2 + 160) )
    ;
  if ( unk_4F072C8 )
    v2 = *(_QWORD *)(v2 + 168);
  if ( (**(_BYTE **)(v2 + 176) & 1) != 0 )
  {
    v4 = *(_QWORD *)(v2 + 168);
    if ( (*(_BYTE *)(v2 + 161) & 0x10) != 0 )
      v4 = *(_QWORD *)(v4 + 96);
    while ( v4 )
    {
      if ( !(unsigned int)sub_621060(v4, a1) )
      {
        *a2 = v4;
        return 1;
      }
      v4 = *(_QWORD *)(v4 + 120);
    }
  }
  return 0;
}
