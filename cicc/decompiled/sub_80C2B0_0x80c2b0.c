// Function: sub_80C2B0
// Address: 0x80c2b0
//
void __fastcall sub_80C2B0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rax
  char v3; // di

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v2 = *(_QWORD *)(a1 + 168);
  v3 = *(_BYTE *)(v2 + 18) & 0x7F;
  if ( (*(_BYTE *)(v2 + 20) & 0x40) != 0 && unk_4D04850 )
  {
    v3 = *(_BYTE *)(v2 + 18) & 0x7E | 1;
LABEL_7:
    sub_80C190(v3, a2);
    return;
  }
  if ( (*(_BYTE *)(v2 + 18) & 0x7F) != 0 )
    goto LABEL_7;
}
