// Function: sub_2FF1F10
// Address: 0x2ff1f10
//
__int64 __fastcall sub_2FF1F10(__int64 a1)
{
  unsigned int v2; // r8d

  if ( (_DWORD)qword_5029648 == 1 )
    return 1;
  if ( (_DWORD)qword_5029648 == 2 )
    return 0;
  if ( (_DWORD)qword_5029648 )
    BUG();
  LOBYTE(v2) = (unsigned int)sub_2FF0570(a1) != 0;
  return v2;
}
