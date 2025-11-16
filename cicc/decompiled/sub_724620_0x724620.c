// Function: sub_724620
// Address: 0x724620
//
FILE *__fastcall sub_724620(char *filename, int a2, int a3, int *a4)
{
  char *v7; // rsi

  if ( sub_720EA0(filename) )
  {
    if ( a3 )
    {
      v7 = "w+b";
      if ( !a2 )
        v7 = "w+";
    }
    else
    {
      v7 = "wb";
      if ( !a2 )
        v7 = "w";
    }
    return sub_7244D0(filename, v7, a4);
  }
  else
  {
    sub_720D70(a4);
    *a4 |= 0x10u;
    return 0;
  }
}
