// Function: sub_724840
// Address: 0x724840
//
char *__fastcall sub_724840(int a1, const char *a2)
{
  __int64 v3; // rdi
  char *v4; // rax
  char *v6; // rax

  v3 = strlen(a2) + 1;
  if ( a1 )
  {
    if ( unk_4F073B8 == a1 )
      v4 = (char *)sub_7247C0(v3);
    else
      v4 = (char *)sub_822B10(v3);
    return strcpy(v4, a2);
  }
  else
  {
    v6 = (char *)sub_823970(v3);
    return strcpy(v6, a2);
  }
}
