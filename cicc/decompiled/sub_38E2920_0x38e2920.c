// Function: sub_38E2920
// Address: 0x38e2920
//
void __fastcall sub_38E2920(__int64 a1, unsigned int a2)
{
  __int16 v2; // bx
  bool v3; // cc

  v2 = a2;
  sub_38E27A0(a1);
  if ( (unsigned int)sub_38E2700(a1) != 3 )
  {
    v3 = a2 <= 2;
    if ( a2 != 2 )
      goto LABEL_3;
LABEL_10:
    v2 = 16;
    goto LABEL_6;
  }
  if ( !a2 )
    goto LABEL_6;
  sub_38E28A0(a1, 0);
  v3 = a2 <= 2;
  if ( a2 == 2 )
    goto LABEL_10;
LABEL_3:
  if ( v3 )
  {
    if ( a2 )
      v2 = 8;
  }
  else
  {
    v2 = 24;
  }
LABEL_6:
  *(_WORD *)(a1 + 12) = *(_WORD *)(a1 + 12) & 0xFFE7 | v2;
}
