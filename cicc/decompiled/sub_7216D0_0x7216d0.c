// Function: sub_7216D0
// Address: 0x7216d0
//
__int64 sub_7216D0()
{
  int v0; // eax
  __int64 result; // rax
  char *v2; // rsi

  if ( !qword_4F07510 )
    return 0;
  v0 = fileno(qword_4F07510);
  if ( !isatty(v0) )
    return 0;
  v2 = getenv("TERM");
  if ( !v2 || (result = 1, !strcmp(v2, "dumb")) )
  {
    dword_4F073C8 = 0;
    return 1;
  }
  return result;
}
