// Function: sub_263F470
// Address: 0x263f470
//
void __fastcall sub_263F470(unsigned int *a1, unsigned int *a2)
{
  unsigned int *v3; // rbx
  unsigned int v4; // ecx
  unsigned int v5; // edx
  unsigned int *v6; // rax
  unsigned int *v7; // rsi
  unsigned int *v8; // rsi

  if ( (char *)a2 - (char *)a1 <= 64 )
  {
    sub_263F170(a1, a2);
  }
  else
  {
    v3 = a1 + 16;
    sub_263F170(a1, a1 + 16);
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
