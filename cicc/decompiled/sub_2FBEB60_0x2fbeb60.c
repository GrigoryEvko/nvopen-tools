// Function: sub_2FBEB60
// Address: 0x2fbeb60
//
char *__fastcall sub_2FBEB60(char *a1, char *a2, char *a3)
{
  char *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r10
  char *v7; // rdx
  __int64 i; // rcx
  __int64 v9; // rsi
  char *v10; // rcx
  char *v11; // rsi
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r11
  char *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  if ( a2 == a3 )
    return a1;
  v3 = &a1[a3 - a2];
  v4 = (a3 - a1) >> 3;
  v5 = (a2 - a1) >> 3;
  if ( v5 != v4 - v5 )
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
          v9 = *(_QWORD *)&a1[8 * i];
          *(_QWORD *)&a1[8 * i] = *(_QWORD *)&v7[8 * i];
          *(_QWORD *)&v7[8 * i] = v9;
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
        v10 = &a1[8 * v4];
        a1 = &v10[-8 * v6];
        if ( v5 > 0 )
        {
          v11 = &v10[-8 * v6];
          v12 = 0;
          do
          {
            v13 = *((_QWORD *)v11 - 1);
            v14 = *((_QWORD *)v10 - 1);
            ++v12;
            v11 -= 8;
            v10 -= 8;
            *(_QWORD *)v11 = v14;
            *(_QWORD *)v10 = v13;
          }
          while ( v5 != v12 );
          a1 -= 8 * v5;
        }
        v5 = v4 % v6;
        if ( !(v4 % v6) )
          return v3;
      }
    }
    return v3;
  }
  v16 = a2;
  do
  {
    v17 = *(_QWORD *)v16;
    v18 = *(_QWORD *)a1;
    a1 += 8;
    v16 += 8;
    *((_QWORD *)a1 - 1) = v17;
    *((_QWORD *)v16 - 1) = v18;
  }
  while ( a2 != a1 );
  return a2;
}
