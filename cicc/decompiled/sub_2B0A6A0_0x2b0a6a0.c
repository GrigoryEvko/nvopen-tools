// Function: sub_2B0A6A0
// Address: 0x2b0a6a0
//
char *__fastcall sub_2B0A6A0(char *a1, char *a2, char *a3)
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
  char *v14; // rcx
  char *v15; // rsi
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rbx
  char *v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rdx

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
    v20 = a2;
    do
    {
      v21 = *(_QWORD *)v20;
      v22 = *(_QWORD *)v4;
      v4 += 16;
      v20 += 16;
      *((_QWORD *)v4 - 2) = v21;
      LODWORD(v21) = *((_DWORD *)v20 - 2);
      *((_QWORD *)v20 - 2) = v22;
      LODWORD(v22) = *((_DWORD *)v4 - 2);
      *((_DWORD *)v4 - 2) = v21;
      *((_DWORD *)v20 - 2) = v22;
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
        v15 = &v14[-16 * v8];
        v16 = 0;
        do
        {
          v17 = *((_QWORD *)v15 - 2);
          v18 = *((_QWORD *)v14 - 2);
          ++v16;
          v15 -= 16;
          v14 -= 16;
          *(_QWORD *)v15 = v18;
          LODWORD(v18) = *((_DWORD *)v14 + 2);
          *(_QWORD *)v14 = v17;
          LODWORD(v17) = *((_DWORD *)v15 + 2);
          *((_DWORD *)v15 + 2) = v18;
          *((_DWORD *)v14 + 2) = v17;
        }
        while ( v7 != v16 );
        v4 -= 16 * v7;
      }
      v7 = v6 % v8;
      if ( !(v6 % v8) )
        return v5;
    }
  }
  return v5;
}
