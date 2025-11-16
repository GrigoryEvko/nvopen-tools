// Function: sub_7244D0
// Address: 0x7244d0
//
FILE *__fastcall sub_7244D0(char *filename, char *modes, int *a3)
{
  const char *v4; // rax
  FILE *v5; // r13
  int v7; // edx
  int v8; // ecx
  int v9; // eax
  int v10; // ecx
  bool v11; // zf
  int v12; // eax
  int v13; // eax
  FILE *v14; // rdi

  sub_720D70(a3);
  if ( *filename )
  {
    v4 = (const char *)sub_7212A0((__int64)filename);
    v5 = fopen(v4, modes);
    if ( v5 )
    {
      if ( !(unsigned int)sub_7244C0((__int64)filename) )
      {
        v11 = !sub_721580(filename);
        v12 = *a3;
        if ( v11 )
          v13 = v12 | 4;
        else
          v13 = v12 | 8;
        *a3 = v13;
        v14 = v5;
        v5 = 0;
        fclose(v14);
      }
      return v5;
    }
    else
    {
      v7 = *__errno_location();
      v8 = *a3;
      v9 = *a3 | 1;
      a3[1] = v7;
      v10 = v8 | 2;
      if ( v7 != 2 )
        v9 = v10;
      *a3 = v9;
      return 0;
    }
  }
  else
  {
    *a3 |= 0x10u;
    return 0;
  }
}
