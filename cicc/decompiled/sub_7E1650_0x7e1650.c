// Function: sub_7E1650
// Address: 0x7e1650
//
void __fastcall sub_7E1650(const char **a1)
{
  char *v1; // r12
  char *v2; // rax
  const char *i; // rdi
  char *v4; // rax

  v1 = (char *)*a1;
  if ( *a1 && strchr(*a1, 92) )
  {
    v2 = sub_7E1620(v1);
    *a1 = v2;
    for ( i = v2; ; i = v4 + 1 )
    {
      v4 = strchr(i, 92);
      if ( !v4 )
        break;
      *v4 = 95;
    }
  }
}
