// Function: sub_7BE800
// Address: 0x7be800
//
__int64 __fastcall sub_7BE800(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  if ( word_4F06418[0] == (_WORD)a1 )
  {
    sub_7B8B50(a1, a2, a3, a4, a5, a6);
    return 1;
  }
  else
  {
    *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
    return 0;
  }
}
