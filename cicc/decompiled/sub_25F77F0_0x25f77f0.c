// Function: sub_25F77F0
// Address: 0x25f77f0
//
char *__fastcall sub_25F77F0(char *a1, char *a2, char *a3)
{
  char *v4; // r9
  char *v5; // r10
  signed __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 *v9; // rcx
  __int64 *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r11
  char *v16; // rcx
  __int64 *v17; // rsi
  __int64 *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v23; // r11
  char *v25; // rax
  char *v26; // rdx
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rdi
  __int64 v30; // rcx

  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v4 = a1;
  v5 = &a1[a3 - a2];
  v6 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 3);
  v7 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
  if ( v7 == v6 - v7 )
  {
    v25 = a1;
    v26 = a2;
    do
    {
      v27 = *(_QWORD *)v26;
      v28 = *(_QWORD *)v25;
      v25 += 24;
      v26 += 24;
      v29 = *((_QWORD *)v25 - 2);
      v30 = *((_QWORD *)v25 - 1);
      *((_QWORD *)v25 - 3) = v27;
      *((_QWORD *)v25 - 2) = *((_QWORD *)v26 - 2);
      *((_QWORD *)v25 - 1) = *((_QWORD *)v26 - 1);
      *((_QWORD *)v26 - 3) = v28;
      *((_QWORD *)v26 - 2) = v29;
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
      v9 = (__int64 *)&v4[24 * v7];
      if ( v8 > 0 )
      {
        v10 = (__int64 *)v4;
        v11 = 0;
        do
        {
          v12 = *v9;
          v13 = *v10;
          ++v11;
          v10 += 3;
          v14 = *(v10 - 2);
          v15 = *(v10 - 1);
          v9 += 3;
          *(v10 - 3) = v12;
          *(v10 - 2) = *(v9 - 2);
          *(v10 - 1) = *(v9 - 1);
          *(v9 - 3) = v13;
          *(v9 - 2) = v14;
          *(v9 - 1) = v15;
        }
        while ( v8 != v11 );
        v4 += 24 * v8;
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
        v16 = &v4[24 * v6];
        v4 = &v16[-24 * v8];
        if ( v7 > 0 )
        {
          v17 = (__int64 *)(v4 - 24);
          v18 = (__int64 *)(v16 - 24);
          v19 = 0;
          do
          {
            v20 = *v18;
            v21 = *v17;
            ++v19;
            v17 -= 3;
            v22 = v17[4];
            v23 = v17[5];
            v18 -= 3;
            v17[3] = v20;
            v17[4] = v18[4];
            v17[5] = v18[5];
            v18[3] = v21;
            v18[4] = v22;
            v18[5] = v23;
          }
          while ( v7 != v19 );
          v4 -= 24 * v7;
        }
        v7 = v6 % v8;
        if ( !(v6 % v8) )
          return v5;
      }
    }
    return v5;
  }
}
