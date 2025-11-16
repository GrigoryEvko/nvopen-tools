// Function: sub_F797F0
// Address: 0xf797f0
//
char *__fastcall sub_F797F0(char *a1, char *a2, char *a3)
{
  char *v3; // r11
  char *v4; // r10
  char *v5; // r11
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  char *v9; // rcx
  char *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // rbx
  __int64 v15; // rdx
  char *v16; // rcx
  char *v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rbx
  __int64 v21; // rbx
  __int64 v22; // rdx
  char *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 == a3 )
    return a1;
  v5 = &a1[a3 - a2];
  v6 = (a3 - a1) >> 4;
  v7 = (a2 - a1) >> 4;
  if ( v7 == v6 - v7 )
  {
    v24 = a2;
    do
    {
      v25 = *(_QWORD *)v24;
      v26 = *(_QWORD *)v4;
      v4 += 16;
      v24 += 16;
      *((_QWORD *)v4 - 2) = v25;
      v27 = *((_QWORD *)v24 - 1);
      *((_QWORD *)v24 - 2) = v26;
      v28 = *((_QWORD *)v4 - 1);
      *((_QWORD *)v4 - 1) = v27;
      *((_QWORD *)v24 - 1) = v28;
    }
    while ( a2 != v4 );
    return a2;
  }
  v8 = v6 - v7;
  if ( v7 >= v6 - v7 )
    goto LABEL_12;
  while ( 1 )
  {
    v9 = &v4[16 * v7];
    if ( v8 > 0 )
    {
      v10 = v4;
      v11 = 0;
      do
      {
        v12 = *(_QWORD *)v10;
        v13 = *(_QWORD *)v9;
        ++v11;
        v10 += 16;
        v9 += 16;
        *((_QWORD *)v10 - 2) = v13;
        v14 = *((_QWORD *)v9 - 1);
        *((_QWORD *)v9 - 2) = v12;
        v15 = *((_QWORD *)v10 - 1);
        *((_QWORD *)v10 - 1) = v14;
        *((_QWORD *)v9 - 1) = v15;
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
        v17 = &v16[-16 * v8];
        v18 = 0;
        do
        {
          v19 = *((_QWORD *)v17 - 2);
          v20 = *((_QWORD *)v16 - 2);
          ++v18;
          v17 -= 16;
          v16 -= 16;
          *(_QWORD *)v17 = v20;
          v21 = *((_QWORD *)v16 + 1);
          *(_QWORD *)v16 = v19;
          v22 = *((_QWORD *)v17 + 1);
          *((_QWORD *)v17 + 1) = v21;
          *((_QWORD *)v16 + 1) = v22;
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
