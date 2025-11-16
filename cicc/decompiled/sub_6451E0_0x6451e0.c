// Function: sub_6451E0
// Address: 0x6451e0
//
unsigned __int64 __fastcall sub_6451E0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 124) & 0x20) != 0 )
  {
    v2 = 5;
    if ( unk_4F04C48 != -1 && (*(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) & 6) == 0 )
      v2 = (unsigned __int8)(5 - (*(_BYTE *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 10) & 1));
    sub_684AA0(v2, 1805, a1 + 72);
    *(_QWORD *)(a1 + 120) &= 0xFFFFFFDFFFFFFF80LL;
    return 0xFFFFFFDFFFFFFF80LL;
  }
  return result;
}
