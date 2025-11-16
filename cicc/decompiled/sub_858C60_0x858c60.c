// Function: sub_858C60
// Address: 0x858c60
//
char *__fastcall sub_858C60(unsigned __int64 a1, unsigned int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  char *result; // rax

  unk_4D03D20 = 1;
  v6 = (unsigned int)dword_4D0493C;
  dword_4D03D1C = dword_4D0493C;
  if ( qword_4F076D0 )
    sub_858BD0(a1, a2, (unsigned int)dword_4D0493C, a4, a5, a6);
  while ( (unsigned __int16)sub_7B8B50(a1, a2, v6, a4, a5, a6) != 9 )
    ;
  result = (char *)&qword_4F06440;
  if ( qword_4F06440 )
  {
    if ( dword_4D0493C )
    {
      sub_7B1260();
      result = (char *)&qword_4D04908;
      if ( qword_4D04908 )
        return sub_7AF460((char *)1, (__int64)a2);
    }
    else
    {
      result = (char *)&qword_4D04908;
      if ( qword_4D04908 )
        return sub_7AF460((char *)1, (__int64)a2);
    }
  }
  return result;
}
