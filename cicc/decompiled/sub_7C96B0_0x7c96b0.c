// Function: sub_7C96B0
// Address: 0x7c96b0
//
__int64 __fastcall sub_7C96B0(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8

  if ( word_4F06418[0] != 9 )
  {
    if ( (_DWORD)a1
      || (a2 = &dword_4F063F8, a1 = 14, sub_6851C0(0xEu, &dword_4F063F8), word_4F06418[0] != 9)
      && (sub_7B8B50(0xEu, &dword_4F063F8, a3, a4, a5, a6), word_4F06418[0] != 9) )
    {
      do
        sub_7B8B50(a1, a2, a3, a4, a5, a6);
      while ( word_4F06418[0] != 9 );
    }
  }
  sub_7B8B50(a1, a2, a3, a4, a5, a6);
  unk_4D03D20 = 0;
  sub_7B8260();
  return sub_863FC0(a1, a2, v6, v7, v8);
}
