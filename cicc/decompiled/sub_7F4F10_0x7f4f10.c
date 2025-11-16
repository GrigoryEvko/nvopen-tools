// Function: sub_7F4F10
// Address: 0x7f4f10
//
_QWORD **sub_7F4F10()
{
  _QWORD **result; // rax

  qword_4F18A20 = 0;
  dword_4D03F8C = 0;
  qword_4F18A18 = 0;
  unk_4D03F88 = 1;
  qword_4F18A10 = 0;
  unk_4D03F78 = 0;
  unk_4D03F70 = 0;
  *(_QWORD *)dword_4D03F38 = *(_QWORD *)&dword_4F077C8;
  unk_4D03F84 = 1;
  byte_4D03F80[0] = sub_622A10(unk_4F069C8, unk_4F069C0, 1);
  sub_808580();
  sub_7DF550();
  result = (_QWORD **)dword_4D04380;
  if ( dword_4D04380 )
    return sub_76FF40();
  return result;
}
