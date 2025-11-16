// Function: sub_8D19F0
// Address: 0x8d19f0
//
__int64 __fastcall sub_8D19F0(__int64 a1)
{
  char v2; // al
  __int64 *v3; // rax
  __int64 v4; // rax
  char v5; // al

  if ( *(_BYTE *)(a1 + 173) != 12 )
    return 0;
  v2 = *(_BYTE *)(a1 + 176);
  if ( !v2 )
    return 1;
  if ( v2 != 1 )
    return 0;
  do
  {
    v3 = sub_72E9A0(a1);
    if ( (*((_BYTE *)v3 + 27) & 2) == 0 )
      break;
    if ( *((_BYTE *)v3 + 24) != 1 )
      break;
    if ( (v3[7] & 0xFD) != 5 )
      break;
    v4 = v3[9];
    if ( *(_BYTE *)(v4 + 24) != 2 )
      break;
    a1 = *(_QWORD *)(v4 + 56);
    if ( *(_BYTE *)(a1 + 173) != 12 )
      break;
    v5 = *(_BYTE *)(a1 + 176);
    if ( !v5 )
      return 1;
  }
  while ( v5 == 1 );
  return 0;
}
