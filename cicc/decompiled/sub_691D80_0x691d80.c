// Function: sub_691D80
// Address: 0x691d80
//
char __fastcall sub_691D80(__int64 a1, _DWORD *a2)
{
  _DWORD *v2; // rax

  v2 = &dword_4F04C44;
  if ( dword_4F04C44 == -1 )
  {
    v2 = (_DWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
    if ( (*((_BYTE *)v2 + 6) & 6) == 0 && *((_BYTE *)v2 + 4) != 12 )
    {
      LOBYTE(v2) = qword_4D03C50;
      if ( !qword_4D03C50 || (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x40) == 0 )
        LOBYTE(v2) = sub_691790(a1, 0, a2);
    }
  }
  return (char)v2;
}
