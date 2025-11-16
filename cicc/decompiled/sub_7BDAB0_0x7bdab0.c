// Function: sub_7BDAB0
// Address: 0x7bdab0
//
__int64 __fastcall sub_7BDAB0(_DWORD *a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax

  if ( dword_4F077C4 == 2 )
  {
    if ( *a1 )
    {
      *a1 = 0;
      dword_4F0664C = ++dword_4F06650[0];
      ++word_4F063FC[0];
      ++WORD2(qword_4F063F0);
      result = 55;
    }
    else
    {
      result = sub_7B8B50((unsigned __int64)a1, a2, a3, a4, a5, a6);
      if ( (_WORD)result == 146 )
      {
        *a1 = 1;
        --WORD2(qword_4F063F0);
        result = 55;
      }
    }
    word_4F06418[0] = result;
  }
  else
  {
    result = sub_7B8B50((unsigned __int64)a1, a2, a3, a4, a5, a6);
    word_4F06418[0] = result;
  }
  return result;
}
