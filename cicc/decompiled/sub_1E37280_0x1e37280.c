// Function: sub_1E37280
// Address: 0x1e37280
//
char *__fastcall sub_1E37280(char *a1, char *a2, char *a3)
{
  char *v4; // r9
  char *v5; // r10
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 *v9; // rcx
  char *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r11
  __int64 v13; // rbx
  __int64 v14; // r11
  char *v15; // rcx
  __int64 *v16; // rsi
  __int64 *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r11
  __int64 v20; // rbx
  __int64 v21; // r11
  char *v23; // rax
  char *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rdi
  __int64 v27; // rcx

  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v4 = a1;
  v5 = &a1[a3 - a2];
  v6 = (a3 - a1) >> 4;
  v7 = (a2 - a1) >> 4;
  if ( v7 == v6 - v7 )
  {
    v23 = a1;
    v24 = a2;
    do
    {
      v25 = *(_QWORD *)v23;
      v26 = *(_QWORD *)v24;
      v23 += 16;
      v24 += 16;
      *((_QWORD *)v23 - 2) = v26;
      *((_QWORD *)v24 - 2) = v25;
      v27 = *((_QWORD *)v24 - 1);
      *((_QWORD *)v24 - 1) = *((_QWORD *)v23 - 1);
      *((_QWORD *)v23 - 1) = v27;
    }
    while ( a2 != v23 );
    return a2;
  }
  else
  {
    v8 = v6 - v7;
    if ( v7 >= v6 - v7 )
      goto LABEL_12;
    while ( 1 )
    {
      v9 = (__int64 *)&v4[16 * v7];
      if ( v8 > 0 )
      {
        v10 = v4;
        v11 = 0;
        do
        {
          v12 = *(_QWORD *)v10;
          v13 = *v9;
          ++v11;
          v10 += 16;
          v9 += 2;
          *((_QWORD *)v10 - 2) = v13;
          *(v9 - 2) = v12;
          v14 = *(v9 - 1);
          *(v9 - 1) = *((_QWORD *)v10 - 1);
          *((_QWORD *)v10 - 1) = v14;
        }
        while ( v8 != v11 );
        v4 += 16 * v8;
      }
      if ( !(v6 % v7) )
        break;
      v8 = v7;
      v7 -= v6 % v7;
      while ( 1 )
      {
        v6 = v8;
        v8 -= v7;
        if ( v7 < v8 )
          break;
LABEL_12:
        v15 = &v4[16 * v6];
        v4 = &v15[-16 * v8];
        if ( v7 > 0 )
        {
          v16 = (__int64 *)(v4 - 16);
          v17 = (__int64 *)(v15 - 16);
          v18 = 0;
          do
          {
            v19 = *v16;
            v20 = *v17;
            ++v18;
            v16 -= 2;
            v17 -= 2;
            v16[2] = v20;
            v17[2] = v19;
            v21 = v17[3];
            v17[3] = v16[3];
            v16[3] = v21;
          }
          while ( v7 != v18 );
          v4 -= 16 * v7;
        }
        v7 = v6 % v8;
        if ( !(v6 % v8) )
          return v5;
      }
    }
    return v5;
  }
}
