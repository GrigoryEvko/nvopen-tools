// Function: sub_8EE7D0
// Address: 0x8ee7d0
//
unsigned __int64 __fastcall sub_8EE7D0(char *a1, int a2)
{
  int v3; // edi
  unsigned __int64 result; // rax
  int v6; // edi
  char v7; // si
  _BYTE *i; // rdx
  unsigned __int64 v9; // rax
  char v10; // cl
  char v11; // cl
  int v12; // ecx
  char v13; // cl

  v3 = a2 + 14;
  if ( a2 + 7 >= 0 )
    v3 = a2 + 7;
  result = (unsigned __int64)(unsigned __int8)*a1 >> 1;
  v6 = v3 >> 3;
  *a1 = result;
  v7 = result;
  if ( a2 > 8 )
  {
    for ( i = a1 + 1; ; ++i )
    {
      v9 = (unsigned __int64)(unsigned __int8)*i >> 1;
      v10 = *i << 7;
      *i = v9;
      v11 = v7 | v10;
      v7 = v9;
      *(i - 1) = v11;
      if ( v6 <= 2 - ((int)a1 + 1) + (int)i )
        break;
    }
    result = (unsigned int)(v6 - 2) + 1LL;
    a1 += result;
    v7 = *a1;
  }
  LOBYTE(v12) = 0x80;
  if ( (a2 & 7) != 0 )
  {
    v13 = a2 % 8 - 1;
    result = (unsigned int)(1 << v13);
    v12 = 1 << v13;
  }
  *a1 = v7 | v12;
  return result;
}
