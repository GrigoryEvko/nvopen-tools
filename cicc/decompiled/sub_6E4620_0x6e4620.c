// Function: sub_6E4620
// Address: 0x6e4620
//
__int64 __fastcall sub_6E4620(_BYTE *a1, __int64 a2)
{
  __int64 result; // rax
  char i; // dl

  result = (__int64)&dword_4F077C4;
  if ( dword_4F077C4 == 2 && a1[16] )
  {
    result = *(_QWORD *)a1;
    for ( i = *(_BYTE *)(*(_QWORD *)a1 + 140LL); i == 12; i = *(_BYTE *)(result + 140) )
      result = *(_QWORD *)(result + 160);
    if ( i )
    {
      result = qword_4D03C50;
      if ( (*(_BYTE *)(qword_4D03C50 + 18LL) & 0x40) == 0 )
      {
        result = dword_4F04C64;
        if ( dword_4F04C64 != -1 )
        {
          result = qword_4F04C68[0] + 776LL * dword_4F04C64;
          if ( (*(_BYTE *)(result + 7) & 2) != 0 && (*(_BYTE *)(a2 + 18) & 0x40) == 0 )
          {
            result = sub_7BEFD0(a2, a1 + 24);
            a1[20] |= 4u;
          }
        }
      }
    }
  }
  return result;
}
