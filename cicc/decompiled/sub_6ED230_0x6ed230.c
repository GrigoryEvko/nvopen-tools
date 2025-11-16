// Function: sub_6ED230
// Address: 0x6ed230
//
_BOOL8 __fastcall sub_6ED230(_BYTE *a1)
{
  __int64 v2; // rax
  char i; // dl

  if ( a1[17] == 2 )
    return 1;
  if ( sub_6ED0A0((__int64)a1) || a1[17] == 3 && (unsigned int)sub_8D3190() || !a1[16] )
    return 1;
  v2 = *(_QWORD *)a1;
  for ( i = *(_BYTE *)(*(_QWORD *)a1 + 140LL); i == 12; i = *(_BYTE *)(v2 + 140) )
    v2 = *(_QWORD *)(v2 + 160);
  return i == 0;
}
