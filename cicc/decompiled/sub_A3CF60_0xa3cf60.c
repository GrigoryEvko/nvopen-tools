// Function: sub_A3CF60
// Address: 0xa3cf60
//
char *__fastcall sub_A3CF60(char *a1, char *a2, char *a3)
{
  char *v4; // r9
  char *v5; // r10
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  char *v9; // rcx
  char *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r11
  __int64 v13; // rbx
  char *v14; // rcx
  char *v15; // rsi
  char *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // r11
  __int64 v19; // rbx
  char *v21; // rax
  char *v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rcx

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
    v21 = a1;
    v22 = a2;
    do
    {
      v23 = *(_QWORD *)v22;
      v24 = *(_QWORD *)v21;
      v21 += 16;
      v22 += 16;
      *((_QWORD *)v21 - 2) = v23;
      LODWORD(v23) = *((_DWORD *)v22 - 2);
      *((_QWORD *)v22 - 2) = v24;
      LODWORD(v24) = *((_DWORD *)v21 - 2);
      *((_DWORD *)v21 - 2) = v23;
      *((_DWORD *)v22 - 2) = v24;
    }
    while ( a2 != v21 );
    return a2;
  }
  else
  {
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
          LODWORD(v13) = *((_DWORD *)v9 - 2);
          *((_QWORD *)v9 - 2) = v12;
          LODWORD(v12) = *((_DWORD *)v10 - 2);
          *((_DWORD *)v10 - 2) = v13;
          *((_DWORD *)v9 - 2) = v12;
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
        v14 = &v4[16 * v6];
        v4 = &v14[-16 * v8];
        if ( v7 > 0 )
        {
          v15 = v4 - 16;
          v16 = v14 - 16;
          v17 = 0;
          do
          {
            v18 = *(_QWORD *)v15;
            v19 = *(_QWORD *)v16;
            ++v17;
            v15 -= 16;
            v16 -= 16;
            *((_QWORD *)v15 + 2) = v19;
            LODWORD(v19) = *((_DWORD *)v16 + 6);
            *((_QWORD *)v16 + 2) = v18;
            LODWORD(v18) = *((_DWORD *)v15 + 6);
            *((_DWORD *)v15 + 6) = v19;
            *((_DWORD *)v16 + 6) = v18;
          }
          while ( v7 != v17 );
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
