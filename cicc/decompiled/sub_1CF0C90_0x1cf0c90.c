// Function: sub_1CF0C90
// Address: 0x1cf0c90
//
__int64 __fastcall sub_1CF0C90(
        char ***a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  char **v9; // r12
  char **v11; // rax
  char *v12; // rcx
  char *v13; // rdi
  char *v14; // rax
  signed __int64 v15; // rdx
  char *v16; // rdx
  char *v17; // rsi
  char **v18; // rax
  __int64 v19; // rdi
  int v20; // edx
  int v21; // r8d
  char *v22; // rsi
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  __int64 v25; // r9
  int v26; // edx
  int v27; // r8d
  char *v28; // rsi
  unsigned int v29; // ecx
  __int64 *v30; // rdx
  __int64 v31; // r9
  int v33; // edx
  int v34; // r10d
  int v35; // edx
  int v36; // r10d

  v9 = *a1;
  v11 = a1[1];
  v12 = (*a1)[1];
  v13 = **a1;
  v14 = *v11;
  v15 = 0xAAAAAAAAAAAAAAABLL * ((v12 - v13) >> 3);
  if ( v15 >> 2 <= 0 )
  {
LABEL_18:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
        {
          v13 = v12;
          goto LABEL_8;
        }
LABEL_26:
        if ( v14 != *(char **)v13 )
          v13 = v12;
        goto LABEL_8;
      }
      if ( v14 == *(char **)v13 )
        goto LABEL_8;
      v13 += 24;
    }
    if ( v14 == *(char **)v13 )
      goto LABEL_8;
    v13 += 24;
    goto LABEL_26;
  }
  v16 = &v13[96 * (v15 >> 2)];
  while ( v14 != *(char **)v13 )
  {
    if ( v14 == *((char **)v13 + 3) )
    {
      v13 += 24;
      break;
    }
    if ( v14 == *((char **)v13 + 6) )
    {
      v13 += 48;
      break;
    }
    if ( v14 == *((char **)v13 + 9) )
    {
      v13 += 72;
      break;
    }
    v13 += 96;
    if ( v13 == v16 )
    {
      v15 = 0xAAAAAAAAAAAAAAABLL * ((v12 - v13) >> 3);
      goto LABEL_18;
    }
  }
LABEL_8:
  v17 = v13 + 24;
  if ( v12 != v13 + 24 )
  {
    memmove(v13, v17, v12 - v17);
    v17 = v9[1];
  }
  v9[1] = v17 - 24;
  v18 = a1[2];
  v19 = (__int64)*a1[1];
  v20 = *((_DWORD *)v18 + 28);
  if ( v20 )
  {
    v21 = v20 - 1;
    v22 = v18[12];
    v23 = (v20 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v24 = (__int64 *)&v22[16 * v23];
    v25 = *v24;
    if ( v19 == *v24 )
    {
LABEL_12:
      *v24 = -16;
      --*((_DWORD *)v18 + 26);
      ++*((_DWORD *)v18 + 27);
      v18 = a1[2];
      v19 = (__int64)*a1[1];
    }
    else
    {
      v33 = 1;
      while ( v25 != -8 )
      {
        v34 = v33 + 1;
        v23 = v21 & (v33 + v23);
        v24 = (__int64 *)&v22[16 * v23];
        v25 = *v24;
        if ( v19 == *v24 )
          goto LABEL_12;
        v33 = v34;
      }
    }
  }
  v26 = *((_DWORD *)v18 + 20);
  if ( v26 )
  {
    v27 = v26 - 1;
    v28 = v18[8];
    v29 = (v26 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v30 = (__int64 *)&v28[16 * v29];
    v31 = *v30;
    if ( v19 == *v30 )
    {
LABEL_15:
      *v30 = -16;
      --*((_DWORD *)v18 + 18);
      ++*((_DWORD *)v18 + 19);
      v19 = (__int64)*a1[1];
    }
    else
    {
      v35 = 1;
      while ( v31 != -8 )
      {
        v36 = v35 + 1;
        v29 = v27 & (v35 + v29);
        v30 = (__int64 *)&v28[16 * v29];
        v31 = *v30;
        if ( v19 == *v30 )
          goto LABEL_15;
        v35 = v36;
      }
    }
  }
  sub_164D160(v19, *(_QWORD *)(v19 + 24 * (1LL - (*(_DWORD *)(v19 + 20) & 0xFFFFFFF))), a2, a3, a4, a5, a6, a7, a8, a9);
  return sub_15F20C0(*a1[1]);
}
