// Function: sub_2D24370
// Address: 0x2d24370
//
__int64 __fastcall sub_2D24370(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8,
        unsigned __int64 a9,
        char a10)
{
  __int64 v11; // rdi
  char v12; // dl
  __int64 v13; // r9
  int v14; // r8d
  int v15; // ebx
  unsigned int i; // eax
  __int64 v17; // r10
  unsigned int v18; // eax
  __int64 v19; // r8
  __int64 v20; // rax
  unsigned __int64 v21; // r8
  int v22; // edx
  _QWORD *v23; // rax
  _QWORD *v24; // rcx
  __int64 v26; // rdx
  unsigned __int64 v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // r10

  v11 = *a1;
  v12 = *(_BYTE *)(v11 + 8) & 1;
  if ( v12 )
  {
    v13 = v11 + 16;
    v14 = 3;
  }
  else
  {
    v19 = *(unsigned int *)(v11 + 24);
    v13 = *(_QWORD *)(v11 + 16);
    if ( !(_DWORD)v19 )
    {
LABEL_43:
      v29 = 96 * v19;
LABEL_44:
      v17 = v13 + v29;
      goto LABEL_10;
    }
    v14 = v19 - 1;
  }
  v15 = 1;
  for ( i = v14
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v14 & v18 )
  {
    v17 = v13 + 96LL * i;
    if ( *(_QWORD *)v17 == a2 && *(_QWORD *)(v17 + 8) == a3 )
      break;
    if ( *(_QWORD *)v17 == -4096 && *(_QWORD *)(v17 + 8) == -4096 )
    {
      if ( !v12 )
      {
        v19 = *(unsigned int *)(v11 + 24);
        goto LABEL_43;
      }
      v29 = 384;
      goto LABEL_44;
    }
    v18 = v15 + i;
    ++v15;
  }
LABEL_10:
  v20 = 384;
  if ( !v12 )
    v20 = 96LL * *(unsigned int *)(v11 + 24);
  LODWORD(v21) = 0;
  if ( v17 == v13 + v20 )
    return (unsigned int)v21;
  v22 = *(_DWORD *)(v17 + 24) >> 1;
  if ( (*(_BYTE *)(v17 + 24) & 1) != 0 )
  {
    v23 = (_QWORD *)(v17 + 96);
    if ( v22 )
    {
      v24 = (_QWORD *)(v17 + 96);
      v23 = (_QWORD *)(v17 + 32);
      goto LABEL_21;
    }
LABEL_17:
    v24 = v23;
    goto LABEL_18;
  }
  v23 = *(_QWORD **)(v17 + 32);
  v24 = &v23[2 * *(unsigned int *)(v17 + 40)];
  if ( !v22 )
  {
    v23 += 2 * *(unsigned int *)(v17 + 40);
    goto LABEL_17;
  }
LABEL_21:
  if ( v23 == v24 )
  {
LABEL_40:
    v23 = v24;
    goto LABEL_18;
  }
  while ( *v23 == -1 )
  {
    if ( v23[1] != -1 )
      goto LABEL_25;
LABEL_39:
    v23 += 2;
    if ( v23 == v24 )
      goto LABEL_40;
  }
  if ( *v23 == -2 && v23[1] == -2 )
    goto LABEL_39;
LABEL_25:
  if ( v23 != v24 )
  {
    v26 = *v23;
LABEL_30:
    v27 = v23[1];
    if ( a10 )
    {
      v21 = a9;
      v28 = a8;
    }
    else
    {
      v28 = qword_4F81350[0];
      v21 = qword_4F81350[1];
    }
    if ( v27 + v26 <= v21 || v27 >= v21 + v28 )
    {
      while ( 1 )
      {
        v23 += 2;
        if ( v23 == v24 )
          break;
        v26 = *v23;
        if ( *v23 == -1 )
        {
          if ( v23[1] != -1 )
            goto LABEL_29;
        }
        else if ( v26 != -2 || v23[1] != -2 )
        {
LABEL_29:
          if ( v23 == v24 )
            break;
          goto LABEL_30;
        }
      }
    }
  }
LABEL_18:
  LOBYTE(v21) = v24 != v23;
  return (unsigned int)v21;
}
