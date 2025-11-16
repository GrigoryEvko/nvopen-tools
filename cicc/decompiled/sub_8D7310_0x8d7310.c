// Function: sub_8D7310
// Address: 0x8d7310
//
_BOOL8 __fastcall sub_8D7310(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r12

  if ( *(_BYTE *)(a1 + 140) != 12 )
    goto LABEL_5;
  do
    a1 = *(_QWORD *)(a1 + 160);
  while ( *(_BYTE *)(a1 + 140) == 12 );
  while ( *(_BYTE *)(a2 + 140) == 12 )
  {
    a2 = *(_QWORD *)(a2 + 160);
LABEL_5:
    ;
  }
  v2 = *(_QWORD *)(a1 + 168);
  v3 = *(_QWORD *)(a2 + 168);
  return sub_72A890(v2) && sub_72A890(v3) || *(_BYTE *)(v2 + 25) == *(_BYTE *)(v3 + 25);
}
