// Function: sub_8B1260
// Address: 0x8b1260
//
__int64 __fastcall sub_8B1260(__int64 a1, char a2, __int64 a3, int a4)
{
  unsigned int v6; // edi

  if ( a4 )
  {
    if ( dword_4F077C4 == 2 && (unsigned int)sub_8D23B0(a1) && (unsigned int)sub_8D3A70(a1) )
      sub_8AD220(a1, 0);
    if ( (unsigned int)sub_8D2310(a1) )
      goto LABEL_4;
    if ( (unsigned int)sub_8D23B0(a1)
      && (!(unsigned int)sub_8D23E0(a1)
       || a2 != 1 && dword_4F04C44 == -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 2) == 0) )
    {
      v6 = 2757;
      goto LABEL_5;
    }
  }
  else if ( (unsigned int)sub_8D2310(a1) )
  {
LABEL_4:
    v6 = 2756;
LABEL_5:
    sub_685360(v6, (_DWORD *)(a3 + 32), a1);
    return 1;
  }
  return 0;
}
