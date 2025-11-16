// Function: sub_668160
// Address: 0x668160
//
_BOOL8 __fastcall sub_668160(__int64 a1, __int64 a2, _DWORD *a3, int a4)
{
  __int64 v6; // rsi
  __int64 v7; // r15

  v6 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  v7 = *(_QWORD *)(v6 + 184);
  if ( a4 )
  {
    if ( (unsigned int)sub_880920(a1, v6, a3) )
      return 1;
    v6 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
  }
  if ( (unsigned int)sub_85ED80(a1, v6) )
  {
    return *(_BYTE *)(v7 + 28) != 3 || *(_QWORD *)(a1 + 64) != *(_QWORD *)(v7 + 32);
  }
  else
  {
    sub_6854C0(551, a2, a1);
    *a3 = 1;
    return 0;
  }
}
