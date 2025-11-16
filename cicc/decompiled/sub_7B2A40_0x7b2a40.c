// Function: sub_7B2A40
// Address: 0x7b2a40
//
__int64 __fastcall sub_7B2A40(unsigned __int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int16 v3; // r9

  if ( !dword_4F17FA0 && (qword_4F06498 > a1 || unk_4F06490 <= a1 || unk_4F06458 || dword_4F17F78) )
  {
    if ( (_DWORD)qword_4F061D0 )
    {
      result = qword_4F061D0;
      *(_QWORD *)a2 = qword_4F061D0;
    }
    else
    {
      return sub_7B0EB0(a1, a2);
    }
  }
  else
  {
    *(_DWORD *)a2 = unk_4F0647C;
    result = *(unsigned int *)&word_4F06480;
    if ( *(_DWORD *)&word_4F06480 && qword_4F06488[*(int *)&word_4F06480 - 1] > a1 )
    {
      result = sub_7AB680(a1);
      *(_WORD *)(a2 + 4) = v3 - result;
    }
    else
    {
      *(_WORD *)(a2 + 4) = a1 - qword_4F06498 + 1 - word_4F06480;
    }
  }
  return result;
}
