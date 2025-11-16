// Function: sub_F70480
// Address: 0xf70480
//
void __fastcall sub_F70480(unsigned __int8 *a1, unsigned __int8 **a2, __int64 a3, char *a4, char a5)
{
  unsigned __int8 **v6; // rbx
  char v7; // r15
  char *v8; // rsi
  unsigned __int8 **v9; // r14
  unsigned __int8 *v10; // rsi
  unsigned __int8 v11; // al

  if ( *a1 > 0x1Cu )
  {
    v6 = a2;
    if ( a4 )
    {
      v7 = *a4;
      v8 = a4;
      if ( (unsigned __int8)*a4 <= 0x1Cu )
        return;
    }
    else
    {
      v8 = (char *)*a2;
      v7 = *v8;
      if ( (unsigned __int8)*v8 <= 0x1Cu )
        return;
    }
    v9 = &v6[a3];
    sub_B45260(a1, (__int64)v8, a5);
    while ( v9 != v6 )
    {
      while ( 1 )
      {
        v10 = *v6;
        v11 = **v6;
        if ( v11 > 0x1Cu && (v7 == v11 || !a4) )
          break;
        if ( v9 == ++v6 )
          return;
      }
      ++v6;
      sub_B45560(a1, (unsigned __int64)v10);
    }
  }
}
