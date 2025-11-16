// Function: sub_855DA0
// Address: 0x855da0
//
__int64 __fastcall sub_855DA0(unsigned __int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  __int64 result; // rax

  v6 = dword_4D04954;
  if ( !dword_4D04954 )
  {
    a1 = 5;
    if ( dword_4D04964 )
      a1 = unk_4F07471;
    a2 = 14;
    sub_684AA0(a1, 0xEu, &dword_4F063F8);
  }
  for ( result = (unsigned int)word_4F06418[0] - 9;
        (unsigned __int16)(word_4F06418[0] - 9) > 1u;
        result = (unsigned int)word_4F06418[0] - 9 )
  {
    sub_7B8B50(a1, (unsigned int *)a2, v6, a4, a5, a6);
  }
  return result;
}
