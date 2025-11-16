// Function: sub_191FBD0
// Address: 0x191fbd0
//
unsigned __int64 __fastcall sub_191FBD0(char *a1, char *a2, char *a3)
{
  char *v3; // r12
  char *v4; // r11
  signed __int64 v5; // rax
  __int64 v6; // rdi
  __int64 v7; // r8
  char *v8; // rcx
  char *v9; // rsi
  __int64 v10; // rdx
  int v11; // r14d
  int v12; // r13d
  int v13; // ebx
  __int64 v14; // r10
  __int64 v15; // r9
  char *v16; // rcx
  char *v17; // rsi
  __int64 v18; // rdx
  int v19; // r14d
  int v20; // r13d
  int v21; // ebx
  __int64 v22; // r10
  __int64 v23; // r9
  char *v25; // rdx
  char *v26; // rax
  int v27; // r10d
  int v28; // r9d
  int v29; // r8d
  __int64 v30; // rdi
  __int64 v31; // rcx

  v3 = a3;
  if ( a1 == a2 )
    return (unsigned __int64)v3;
  v4 = a1;
  if ( a2 != a3 )
  {
    v3 = &a1[a3 - a2];
    v5 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a1) >> 3);
    v6 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 3);
    if ( v6 == v5 - v6 )
    {
      v25 = a2;
      v26 = v4;
      do
      {
        v27 = *(_DWORD *)v25;
        v28 = *(_DWORD *)v26;
        v26 += 24;
        v25 += 24;
        v29 = *((_DWORD *)v26 - 5);
        v30 = *((_QWORD *)v26 - 2);
        *((_DWORD *)v26 - 6) = v27;
        v31 = *((_QWORD *)v26 - 1);
        *((_DWORD *)v26 - 5) = *((_DWORD *)v25 - 5);
        *((_QWORD *)v26 - 2) = *((_QWORD *)v25 - 2);
        *((_QWORD *)v26 - 1) = *((_QWORD *)v25 - 1);
        *((_DWORD *)v25 - 6) = v28;
        *((_DWORD *)v25 - 5) = v29;
        *((_QWORD *)v25 - 2) = v30;
        *((_QWORD *)v25 - 1) = v31;
      }
      while ( a2 != v26 );
      return (unsigned __int64)&v4[8 * ((unsigned __int64)(a2 - 24 - v4) >> 3) + 24];
    }
    else
    {
      v7 = v5 - v6;
      if ( v6 >= v5 - v6 )
        goto LABEL_12;
      while ( 1 )
      {
        v8 = &v4[24 * v6];
        if ( v7 > 0 )
        {
          v9 = v4;
          v10 = 0;
          do
          {
            v11 = *(_DWORD *)v8;
            v12 = *(_DWORD *)v9;
            ++v10;
            v9 += 24;
            v13 = *((_DWORD *)v9 - 5);
            v14 = *((_QWORD *)v9 - 2);
            v8 += 24;
            *((_DWORD *)v9 - 6) = v11;
            v15 = *((_QWORD *)v9 - 1);
            *((_DWORD *)v9 - 5) = *((_DWORD *)v8 - 5);
            *((_QWORD *)v9 - 2) = *((_QWORD *)v8 - 2);
            *((_QWORD *)v9 - 1) = *((_QWORD *)v8 - 1);
            *((_DWORD *)v8 - 6) = v12;
            *((_DWORD *)v8 - 5) = v13;
            *((_QWORD *)v8 - 2) = v14;
            *((_QWORD *)v8 - 1) = v15;
          }
          while ( v7 != v10 );
          v4 += 24 * v7;
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
          v16 = &v4[24 * v5];
          v4 = &v16[-24 * v7];
          if ( v6 > 0 )
          {
            v17 = &v16[-24 * v7];
            v18 = 0;
            do
            {
              v19 = *((_DWORD *)v16 - 6);
              v20 = *((_DWORD *)v17 - 6);
              ++v18;
              v17 -= 24;
              v21 = *((_DWORD *)v17 + 1);
              v22 = *((_QWORD *)v17 + 1);
              v16 -= 24;
              *(_DWORD *)v17 = v19;
              v23 = *((_QWORD *)v17 + 2);
              *((_DWORD *)v17 + 1) = *((_DWORD *)v16 + 1);
              *((_QWORD *)v17 + 1) = *((_QWORD *)v16 + 1);
              *((_QWORD *)v17 + 2) = *((_QWORD *)v16 + 2);
              *(_DWORD *)v16 = v20;
              *((_DWORD *)v16 + 1) = v21;
              *((_QWORD *)v16 + 1) = v22;
              *((_QWORD *)v16 + 2) = v23;
            }
            while ( v6 != v18 );
            v4 -= 24 * v6;
          }
          v6 = v5 % v7;
          if ( !(v5 % v7) )
            return (unsigned __int64)v3;
        }
      }
    }
    return (unsigned __int64)v3;
  }
  return (unsigned __int64)a1;
}
