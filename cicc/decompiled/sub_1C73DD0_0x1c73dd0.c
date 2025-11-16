// Function: sub_1C73DD0
// Address: 0x1c73dd0
//
__int64 __fastcall sub_1C73DD0(__int64 a1)
{
  __int64 v1; // rbx
  unsigned int v2; // r12d

  v1 = *(_QWORD *)(a1 + 8);
  if ( !v1 )
    return 0;
  while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v1) + 16) - 25) > 9u )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return 0;
  }
  v2 = 0;
LABEL_5:
  ++v2;
  while ( 1 )
  {
    v1 = *(_QWORD *)(v1 + 8);
    if ( !v1 )
      return v2;
    if ( (unsigned __int8)(*((_BYTE *)sub_1648700(v1) + 16) - 25) <= 9u )
      goto LABEL_5;
  }
}
