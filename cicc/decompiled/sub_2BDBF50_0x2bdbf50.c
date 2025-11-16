// Function: sub_2BDBF50
// Address: 0x2bdbf50
//
void __fastcall sub_2BDBF50(char *a1, char *a2)
{
  char *v3; // rbx
  char v4; // cl
  char v5; // dl
  char *v6; // rax
  char *v7; // rsi
  char *v8; // rsi

  if ( a2 - a1 <= 16 )
  {
    sub_2BDBE90(a1, a2);
  }
  else
  {
    v3 = a1 + 16;
    sub_2BDBE90(a1, a1 + 16);
    if ( a2 != a1 + 16 )
    {
      do
      {
        while ( 1 )
        {
          v4 = *v3;
          v5 = *(v3 - 1);
          v6 = v3 - 1;
          if ( *v3 < v5 )
            break;
          v8 = v3++;
          *v8 = v4;
          if ( a2 == v3 )
            return;
        }
        do
        {
          v6[1] = v5;
          v7 = v6;
          v5 = *--v6;
        }
        while ( v4 < v5 );
        ++v3;
        *v7 = v4;
      }
      while ( a2 != v3 );
    }
  }
}
