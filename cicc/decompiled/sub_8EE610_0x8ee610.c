// Function: sub_8EE610
// Address: 0x8ee610
//
unsigned __int64 __fastcall sub_8EE610(char *a1, unsigned int a2, int a3, int a4)
{
  unsigned __int64 result; // rax
  int v6; // esi
  int v8; // esi
  int v9; // r11d
  int v10; // edi
  _BYTE *v11; // r9
  int v12; // edx
  int v13; // edi

  result = a2;
  v6 = a2 + 14;
  if ( (int)result + 7 >= 0 )
    v6 = result + 7;
  v8 = v6 >> 3;
  if ( (int)result < a3 )
  {
    if ( (int)result > 0 )
      return (unsigned __int64)memset(a1, 0, (unsigned int)(v8 - 1) + 1LL);
    return result;
  }
  if ( !a3 )
    return result;
  result = (unsigned int)(a3 >> 31) >> 29;
  v9 = a3 / 8;
  v10 = a3 % 8;
  if ( v8 > a3 / 8 )
  {
    result = v9;
    v11 = a1;
    while ( 1 )
    {
      v12 = (int)(unsigned __int8)a1[result] >> v10;
      *v11 = v12;
      if ( v8 - 1 > (int)result )
        *v11 = ((unsigned __int8)a1[result + 1] << (8 - v10)) + v12;
      if ( !a4 )
        goto LABEL_10;
      if ( *v11 == 0xFF )
      {
        ++result;
        *v11++ = 0;
        if ( v8 <= (int)result )
        {
LABEL_16:
          v13 = v8 - v9;
          goto LABEL_17;
        }
      }
      else
      {
        a4 = 0;
        ++*v11;
LABEL_10:
        ++result;
        ++v11;
        if ( v8 <= (int)result )
          goto LABEL_16;
      }
    }
  }
  v13 = 0;
LABEL_17:
  if ( a4 )
  {
    result = v13++;
    a1[result] = 1;
  }
  if ( v8 > v13 )
    return (unsigned __int64)memset(&a1[v13], 0, (unsigned int)(v8 - 1 - v13) + 1LL);
  return result;
}
