// Function: sub_2FF0E00
// Address: 0x2ff0e00
//
void __fastcall sub_2FF0E00(__int64 a1, __int64 a2)
{
  if ( !*(_BYTE *)(a1 + 251) )
    goto LABEL_5;
  if ( (_DWORD)qword_5028A08 != 1 )
  {
    if ( (_DWORD)qword_5028AE8 == 1 )
      sub_2FF0D60(a1);
LABEL_5:
    sub_2FF0CC0(a1, a2);
    return;
  }
  sub_2FF0DA0(a1);
  sub_2FF0D60(a1);
  sub_2FF0CC0(a1, a2);
}
