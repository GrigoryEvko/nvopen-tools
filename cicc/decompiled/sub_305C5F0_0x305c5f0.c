// Function: sub_305C5F0
// Address: 0x305c5f0
//
__int64 __fastcall sub_305C5F0(__int64 a1)
{
  unsigned __int8 v1; // dl
  _QWORD *v3; // rax
  _QWORD *v4; // rax

  if ( (_DWORD)qword_502C568 == 1 )
  {
    if ( !(_BYTE)qword_502C488 && (unsigned int)sub_2FF0570(a1) )
    {
      v3 = (_QWORD *)sub_308A6E0();
      sub_2FF0E80(a1, v3, 0);
    }
    v1 = 0;
  }
  else
  {
    sub_2FF12A0(a1, &unk_5022C2C, 1u);
    if ( !(_BYTE)qword_502C488 && (unsigned int)sub_2FF0570(a1) )
    {
      v4 = (_QWORD *)sub_308A6E0();
      sub_2FF0E80(a1, v4, 0);
    }
    v1 = 1;
  }
  return sub_2FF12A0(a1, &unk_502A48C, v1);
}
