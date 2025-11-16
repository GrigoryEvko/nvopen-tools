// Function: sub_679930
// Address: 0x679930
//
__int64 *__fastcall sub_679930(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *result; // rax
  int v5; // edi

  sub_7B8B50(a1, a2, a3, a4);
  result = (__int64 *)&dword_4F077C4;
  if ( dword_4F077C4 == 2 )
  {
    if ( word_4F06418[0] != 1 || (result = &qword_4D04A00, (unk_4D04A11 & 2) == 0) )
    {
      if ( (a1 & 0x40) == 0 || (v5 = 17409, (_DWORD)a2) )
        v5 = 16385;
      return (__int64 *)sub_7C0F00((unsigned int)a2 | v5, 0);
    }
  }
  return result;
}
