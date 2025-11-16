// Function: sub_2664930
// Address: 0x2664930
//
char *__fastcall sub_2664930(char *a1, char *a2, char *a3)
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
  __int64 v14; // rbx
  __int64 v15; // r11
  char *v16; // rcx
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r11
  __int64 v21; // rbx
  __int64 v22; // rbx
  __int64 v23; // r11
  char *v25; // rax
  char *v26; // rdx
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // rcx

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
    v25 = a1;
    v26 = a2;
    do
    {
      v27 = *(_QWORD *)v26;
      v28 = *(_QWORD *)v25;
      v25 += 16;
      v26 += 16;
      *((_QWORD *)v25 - 2) = v27;
      v29 = *((_QWORD *)v26 - 1);
      *((_QWORD *)v26 - 2) = v28;
      v30 = *((_QWORD *)v25 - 1);
      *((_QWORD *)v25 - 1) = v29;
      *((_QWORD *)v26 - 1) = v30;
    }
    while ( a2 != v25 );
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
          v14 = *(v9 - 1);
          *(v9 - 2) = v12;
          v15 = *((_QWORD *)v10 - 1);
          *((_QWORD *)v10 - 1) = v14;
          *(v9 - 1) = v15;
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
        v16 = &v4[16 * v6];
        v4 = &v16[-16 * v8];
        if ( v7 > 0 )
        {
          v17 = (__int64 *)(v4 - 16);
          v18 = (__int64 *)(v16 - 16);
          v19 = 0;
          do
          {
            v20 = *v17;
            v21 = *v18;
            ++v19;
            v17 -= 2;
            v18 -= 2;
            v17[2] = v21;
            v22 = v18[3];
            v18[2] = v20;
            v23 = v17[3];
            v17[3] = v22;
            v18[3] = v23;
          }
          while ( v7 != v19 );
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
