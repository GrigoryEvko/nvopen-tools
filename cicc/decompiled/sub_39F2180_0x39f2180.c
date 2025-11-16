// Function: sub_39F2180
// Address: 0x39f2180
//
void __fastcall sub_39F2180(__int64 a1, __int64 a2, __int64 a3)
{
  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  if ( (*(_BYTE *)(a2 + 8) & 0x10) != 0 )
  {
    sub_390D5F0(*(_QWORD *)(a1 + 264), a3, 0);
    *(_BYTE *)(a3 + 8) |= 0x10u;
    *(_WORD *)(a3 + 12) &= ~1u;
    if ( *(char *)(a2 + 12) >= 0 )
    {
LABEL_3:
      if ( (*(_BYTE *)(a2 + 8) & 0x20) == 0 )
        return;
LABEL_7:
      sub_390D5F0(*(_QWORD *)(a1 + 264), a3, 0);
      *(_BYTE *)(a3 + 8) |= 0x30u;
      return;
    }
  }
  else if ( *(char *)(a2 + 12) >= 0 )
  {
    goto LABEL_3;
  }
  sub_390D5F0(*(_QWORD *)(a1 + 264), a3, 0);
  *(_WORD *)(a3 + 12) |= 0x80u;
  if ( (*(_BYTE *)(a2 + 8) & 0x20) != 0 )
    goto LABEL_7;
}
