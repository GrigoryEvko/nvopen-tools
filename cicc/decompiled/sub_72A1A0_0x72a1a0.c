// Function: sub_72A1A0
// Address: 0x72a1a0
//
void __fastcall sub_72A1A0(__int64 a1)
{
  char v1; // al
  bool v2; // zf

  if ( *(_BYTE *)(a1 + 173) != 12 )
    goto LABEL_9;
  v1 = *(_BYTE *)(a1 + 176);
  if ( (v1 & 0xFD) != 0 )
  {
    if ( v1 == 13 || v1 == 3 )
      goto LABEL_5;
LABEL_9:
    v2 = *(_QWORD *)(a1 + 144) == 0;
    *(_QWORD *)a1 = 0;
    if ( !v2 )
      *(_QWORD *)(a1 + 8) = 0;
    *(_WORD *)(a1 + 88) &= 0x83FCu;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    sub_72A140(a1);
    goto LABEL_12;
  }
  if ( !v1 )
    *(_QWORD *)(a1 + 40) = 0;
LABEL_5:
  sub_72A140(a1);
  if ( *(_BYTE *)(a1 + 173) != 12 || *(_BYTE *)(a1 + 176) != 2 )
  {
    *(_BYTE *)(a1 + 170) &= ~0x10u;
    *(_QWORD *)a1 = 0;
    return;
  }
LABEL_12:
  *(_BYTE *)(a1 + 170) &= ~0x10u;
}
