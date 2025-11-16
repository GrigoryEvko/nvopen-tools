// Function: sub_371DC50
// Address: 0x371dc50
//
__int64 __fastcall sub_371DC50(char ***a1)
{
  char **v1; // r12
  char **v3; // rax
  char *v4; // rcx
  char *v5; // rdi
  char *v6; // rax
  signed __int64 v7; // rdx
  char *v8; // rdx
  char *v9; // rsi
  char **v10; // rax
  char *v11; // rsi
  __int64 v12; // rdi
  int v13; // edx
  int v14; // r8d
  unsigned int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // r9
  int v18; // edx
  char *v19; // rsi
  int v20; // r8d
  unsigned int v21; // ecx
  __int64 *v22; // rdx
  __int64 v23; // r9
  int v25; // edx
  int v26; // r10d
  int v27; // edx
  int v28; // r10d

  v1 = *a1;
  v3 = a1[1];
  v4 = (*a1)[1];
  v5 = **a1;
  v6 = *v3;
  v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - v5) >> 3);
  if ( v7 >> 2 <= 0 )
  {
LABEL_18:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
        {
          v5 = v4;
          goto LABEL_8;
        }
LABEL_26:
        if ( v6 != *(char **)v5 )
          v5 = v4;
        goto LABEL_8;
      }
      if ( v6 == *(char **)v5 )
        goto LABEL_8;
      v5 += 24;
    }
    if ( v6 == *(char **)v5 )
      goto LABEL_8;
    v5 += 24;
    goto LABEL_26;
  }
  v8 = &v5[96 * (v7 >> 2)];
  while ( v6 != *(char **)v5 )
  {
    if ( v6 == *((char **)v5 + 3) )
    {
      v5 += 24;
      break;
    }
    if ( v6 == *((char **)v5 + 6) )
    {
      v5 += 48;
      break;
    }
    if ( v6 == *((char **)v5 + 9) )
    {
      v5 += 72;
      break;
    }
    v5 += 96;
    if ( v8 == v5 )
    {
      v7 = 0xAAAAAAAAAAAAAAABLL * ((v4 - v5) >> 3);
      goto LABEL_18;
    }
  }
LABEL_8:
  v9 = v5 + 24;
  if ( v4 != v5 + 24 )
  {
    memmove(v5, v9, v4 - v9);
    v9 = v1[1];
  }
  v1[1] = v9 - 24;
  v10 = a1[2];
  v11 = v10[12];
  v12 = (__int64)*a1[1];
  v13 = *((_DWORD *)v10 + 28);
  if ( v13 )
  {
    v14 = v13 - 1;
    v15 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v16 = (__int64 *)&v11[16 * v15];
    v17 = *v16;
    if ( v12 == *v16 )
    {
LABEL_12:
      *v16 = -8192;
      --*((_DWORD *)v10 + 26);
      ++*((_DWORD *)v10 + 27);
      v10 = a1[2];
      v12 = (__int64)*a1[1];
    }
    else
    {
      v25 = 1;
      while ( v17 != -4096 )
      {
        v26 = v25 + 1;
        v15 = v14 & (v25 + v15);
        v16 = (__int64 *)&v11[16 * v15];
        v17 = *v16;
        if ( v12 == *v16 )
          goto LABEL_12;
        v25 = v26;
      }
    }
  }
  v18 = *((_DWORD *)v10 + 20);
  v19 = v10[8];
  if ( v18 )
  {
    v20 = v18 - 1;
    v21 = (v18 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
    v22 = (__int64 *)&v19[16 * v21];
    v23 = *v22;
    if ( *v22 == v12 )
    {
LABEL_15:
      *v22 = -8192;
      --*((_DWORD *)v10 + 18);
      ++*((_DWORD *)v10 + 19);
      v12 = (__int64)*a1[1];
    }
    else
    {
      v27 = 1;
      while ( v23 != -4096 )
      {
        v28 = v27 + 1;
        v21 = v20 & (v27 + v21);
        v22 = (__int64 *)&v19[16 * v21];
        v23 = *v22;
        if ( *v22 == v12 )
          goto LABEL_15;
        v27 = v28;
      }
    }
  }
  sub_BD84D0(v12, *(_QWORD *)(v12 + 32 * (1LL - (*(_DWORD *)(v12 + 4) & 0x7FFFFFF))));
  return sub_B43D60(*a1[1]);
}
