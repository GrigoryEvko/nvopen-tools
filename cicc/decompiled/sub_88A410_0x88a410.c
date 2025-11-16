// Function: sub_88A410
// Address: 0x88a410
//
__int64 __fastcall sub_88A410(__int64 a1, __int64 a2)
{
  _BYTE *v2; // rax
  char v3; // dl
  _BYTE *v4; // rcx

  v2 = &unk_4B7DBA0;
  if ( qword_4F06A7C )
    v2 = &unk_4B7DB40;
  for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
    ;
  while ( 1 )
  {
    v3 = *v2;
    if ( *v2 == 13 )
      break;
    while ( 1 )
    {
      v4 = v2;
      v2 += 24;
      if ( *(_BYTE *)(a1 + 160) == v3 && *((_QWORD *)v2 - 2) == a2 )
        break;
      v3 = *v2;
      if ( *v2 == 13 )
        return 0;
    }
    if ( *((_QWORD *)v4 + 2) )
      return *((_QWORD *)v4 + 2);
  }
  return 0;
}
