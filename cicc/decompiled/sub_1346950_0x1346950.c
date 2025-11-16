// Function: sub_1346950
// Address: 0x1346950
//
__int64 __fastcall sub_1346950(_DWORD *a1, char *a2, char **a3)
{
  __int64 result; // rax
  unsigned int v5; // ecx
  int v6; // eax
  char v7; // bl
  char *v8; // rsi
  int v9; // r11d
  __int64 v10; // r10
  __int64 v11; // rax
  unsigned __int64 v12; // r9
  unsigned __int8 v13; // dl
  char v14; // bl

  if ( *a2 == 46 )
  {
    v5 = 0;
LABEL_11:
    result = 1;
    v7 = a2[1];
    if ( (unsigned __int8)(v7 - 48) <= 9u )
    {
      v8 = a2 + 1;
      v9 = 14;
      v10 = 1;
      v11 = 0;
      do
      {
        v12 = 10 * v10;
        v13 = v7 - 48;
        v11 *= 10;
        v10 *= 10;
        if ( (unsigned __int8)(v7 - 48) <= 9u )
        {
          v7 = *++v8;
          v11 += (char)v13;
          v13 = v7 - 48;
        }
        --v9;
      }
      while ( v9 );
      if ( v13 <= 9u )
      {
        do
          v14 = *++v8;
        while ( (unsigned __int8)(v14 - 48) <= 9u );
      }
      *a1 = (v5 << 16) + (v11 << 16) / v12;
      if ( a3 )
        *a3 = v8;
      return 0;
    }
  }
  else
  {
    result = 1;
    v5 = *a2 - 48;
    if ( (unsigned __int8)v5 <= 9u )
    {
      while ( 1 )
      {
        v6 = *++a2;
        if ( (unsigned __int8)(v6 - 48) > 9u )
          break;
        v5 = v6 + 10 * v5 - 48;
        if ( v5 > 0xFFFF )
          return 1;
      }
      if ( (_BYTE)v6 == 46 )
        goto LABEL_11;
      *a1 = v5 << 16;
      if ( a3 )
        *a3 = a2;
      return 0;
    }
  }
  return result;
}
