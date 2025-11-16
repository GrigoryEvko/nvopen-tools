// Function: sub_27A0CC0
// Address: 0x27a0cc0
//
char *__fastcall sub_27A0CC0(char *a1, char *a2, char *a3)
{
  char *v3; // r12
  char *v4; // r11
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  char *v8; // rcx
  char *v9; // rsi
  __int64 v10; // rdi
  int v11; // r14d
  int v12; // r13d
  __int64 v13; // rbx
  __int64 v14; // r10
  __int64 v15; // rdx
  char *v16; // rcx
  char *v17; // rsi
  __int64 v18; // rdi
  int v19; // r14d
  int v20; // r13d
  __int64 v21; // rbx
  __int64 v22; // r10
  __int64 v23; // rdx
  char *v25; // rax
  int v26; // r9d
  int v27; // r8d
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx

  v3 = a3;
  if ( a1 == a2 )
    return v3;
  v4 = a1;
  if ( a2 != a3 )
  {
    v3 = &a1[a3 - a2];
    v5 = (a3 - a1) >> 5;
    v6 = (a2 - a1) >> 5;
    if ( v6 == v5 - v6 )
    {
      v25 = a2;
      do
      {
        v26 = *(_DWORD *)v25;
        v27 = *(_DWORD *)v4;
        v4 += 32;
        v25 += 32;
        v28 = *((_QWORD *)v4 - 3);
        v29 = *((_QWORD *)v4 - 2);
        *((_DWORD *)v4 - 8) = v26;
        v30 = *((_QWORD *)v4 - 1);
        *((_QWORD *)v4 - 3) = *((_QWORD *)v25 - 3);
        *((_QWORD *)v4 - 2) = *((_QWORD *)v25 - 2);
        *((_QWORD *)v4 - 1) = *((_QWORD *)v25 - 1);
        *((_DWORD *)v25 - 8) = v27;
        *((_QWORD *)v25 - 3) = v28;
        *((_QWORD *)v25 - 2) = v29;
        *((_QWORD *)v25 - 1) = v30;
      }
      while ( a2 != v4 );
      return a2;
    }
    else
    {
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
      while ( 1 )
      {
        v8 = &v4[32 * v6];
        if ( v7 > 0 )
        {
          v9 = v4;
          v10 = 0;
          do
          {
            v11 = *(_DWORD *)v8;
            v12 = *(_DWORD *)v9;
            ++v10;
            v9 += 32;
            v13 = *((_QWORD *)v9 - 3);
            v14 = *((_QWORD *)v9 - 2);
            v8 += 32;
            *((_DWORD *)v9 - 8) = v11;
            v15 = *((_QWORD *)v9 - 1);
            *((_QWORD *)v9 - 3) = *((_QWORD *)v8 - 3);
            *((_QWORD *)v9 - 2) = *((_QWORD *)v8 - 2);
            *((_QWORD *)v9 - 1) = *((_QWORD *)v8 - 1);
            *((_DWORD *)v8 - 8) = v12;
            *((_QWORD *)v8 - 3) = v13;
            *((_QWORD *)v8 - 2) = v14;
            *((_QWORD *)v8 - 1) = v15;
          }
          while ( v7 != v10 );
          v4 += 32 * v7;
        }
        if ( !(v5 % v6) )
          break;
        v7 = v6;
        v6 -= v5 % v6;
        while ( 1 )
        {
          v5 = v7;
          v7 -= v6;
          if ( v6 < v7 )
            break;
LABEL_12:
          v16 = &v4[32 * v5];
          v4 = &v16[-32 * v7];
          if ( v6 > 0 )
          {
            v17 = &v16[-32 * v7];
            v18 = 0;
            do
            {
              v19 = *((_DWORD *)v16 - 8);
              v20 = *((_DWORD *)v17 - 8);
              ++v18;
              v17 -= 32;
              v21 = *((_QWORD *)v17 + 1);
              v22 = *((_QWORD *)v17 + 2);
              v16 -= 32;
              *(_DWORD *)v17 = v19;
              v23 = *((_QWORD *)v17 + 3);
              *((_QWORD *)v17 + 1) = *((_QWORD *)v16 + 1);
              *((_QWORD *)v17 + 2) = *((_QWORD *)v16 + 2);
              *((_QWORD *)v17 + 3) = *((_QWORD *)v16 + 3);
              *(_DWORD *)v16 = v20;
              *((_QWORD *)v16 + 1) = v21;
              *((_QWORD *)v16 + 2) = v22;
              *((_QWORD *)v16 + 3) = v23;
            }
            while ( v6 != v18 );
            v4 -= 32 * v6;
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return v3;
        }
      }
    }
    return v3;
  }
  return a1;
}
