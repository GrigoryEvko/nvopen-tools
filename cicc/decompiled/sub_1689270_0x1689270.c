// Function: sub_1689270
// Address: 0x1689270
//
void __fastcall sub_1689270(void *a1)
{
  if ( !unk_4CD28E1 )
  {
    if ( !unk_4CD28E0 )
      return;
LABEL_5:
    sub_1689130(a1);
    unk_4CD28E0 = 0;
    return;
  }
  sub_1683CD0();
  unk_4CD28E1 = 0;
  if ( unk_4CD28E0 )
    goto LABEL_5;
}
