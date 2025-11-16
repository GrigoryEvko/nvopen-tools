// Function: sub_2C4CF50
// Address: 0x2c4cf50
//
char *__fastcall sub_2C4CF50(char *a1, char *a2, char *a3)
{
  char *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r10
  char *v7; // rsi
  __int64 i; // rcx
  int v9; // edx
  int v10; // r8d
  int v11; // edx
  char *v12; // rcx
  char *v13; // rsi
  __int64 v14; // r8
  int v15; // edx
  int v16; // r11d
  int v17; // r11d
  int v18; // edx
  char *v20; // rax
  int v21; // ecx
  int v22; // edx
  int v23; // ecx
  int v24; // edx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  if ( a2 != a3 )
  {
    v3 = &a1[a3 - a2];
    v4 = (a3 - a1) >> 3;
    v5 = (a2 - a1) >> 3;
    if ( v5 == v4 - v5 )
    {
      v20 = a2;
      do
      {
        v21 = *(_DWORD *)v20;
        v22 = *(_DWORD *)a1;
        a1 += 8;
        v20 += 8;
        *((_DWORD *)a1 - 2) = v21;
        v23 = *((_DWORD *)v20 - 1);
        *((_DWORD *)v20 - 2) = v22;
        v24 = *((_DWORD *)a1 - 1);
        *((_DWORD *)a1 - 1) = v23;
        *((_DWORD *)v20 - 1) = v24;
      }
      while ( a2 != a1 );
      return a2;
    }
    else
    {
      v6 = v4 - v5;
      if ( v5 >= v4 - v5 )
        goto LABEL_12;
      while ( 1 )
      {
        v7 = &a1[8 * v5];
        if ( v6 > 0 )
        {
          for ( i = 0; i != v6; ++i )
          {
            v9 = *(_DWORD *)&a1[8 * i];
            *(_DWORD *)&a1[8 * i] = *(_DWORD *)&v7[8 * i];
            v10 = *(_DWORD *)&v7[8 * i + 4];
            *(_DWORD *)&v7[8 * i] = v9;
            v11 = *(_DWORD *)&a1[8 * i + 4];
            *(_DWORD *)&a1[8 * i + 4] = v10;
            *(_DWORD *)&v7[8 * i + 4] = v11;
          }
          a1 += 8 * v6;
        }
        if ( !(v4 % v5) )
          break;
        v6 = v5;
        v5 -= v4 % v5;
        while ( 1 )
        {
          v4 = v6;
          v6 -= v5;
          if ( v5 < v6 )
            break;
LABEL_12:
          v12 = &a1[8 * v4];
          a1 = &v12[-8 * v6];
          if ( v5 > 0 )
          {
            v13 = &v12[-8 * v6];
            v14 = 0;
            do
            {
              v15 = *((_DWORD *)v13 - 2);
              v16 = *((_DWORD *)v12 - 2);
              ++v14;
              v13 -= 8;
              v12 -= 8;
              *(_DWORD *)v13 = v16;
              v17 = *((_DWORD *)v12 + 1);
              *(_DWORD *)v12 = v15;
              v18 = *((_DWORD *)v13 + 1);
              *((_DWORD *)v13 + 1) = v17;
              *((_DWORD *)v12 + 1) = v18;
            }
            while ( v5 != v14 );
            a1 -= 8 * v5;
          }
          v5 = v4 % v6;
          if ( !(v4 % v6) )
            return v3;
        }
      }
    }
    return v3;
  }
  return a1;
}
