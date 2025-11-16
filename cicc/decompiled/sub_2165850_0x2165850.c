// Function: sub_2165850
// Address: 0x2165850
//
__int64 __fastcall sub_2165850(__int64 a1)
{
  unsigned __int8 v1; // r8
  _QWORD *v3; // rax
  _QWORD *v4; // rax

  if ( dword_4FD26A0 == 1 )
  {
    if ( !byte_4FD25C0 && (unsigned int)sub_1F45DD0(a1) )
    {
      v3 = (_QWORD *)sub_21F9D90();
      sub_1F46490(a1, v3, 1, 1, 0);
    }
    v1 = 0;
  }
  else
  {
    sub_1F46F00(a1, &unk_4FC8A0C, 1, 1, 1u);
    if ( !byte_4FD25C0 && (unsigned int)sub_1F45DD0(a1) )
    {
      v4 = (_QWORD *)sub_21F9D90();
      sub_1F46490(a1, v4, 1, 1, 0);
    }
    v1 = 1;
  }
  return sub_1F46F00(a1, &unk_4FCE24C, 1, 1, v1);
}
