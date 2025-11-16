// Function: sub_2018A60
// Address: 0x2018a60
//
__int64 __fastcall sub_2018A60(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v7; // rcx
  unsigned __int64 v8; // r8
  _QWORD *v9; // r9
  __int64 v10; // r12
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 *v13; // rdx
  __int64 v14; // rdi
  unsigned int v15; // eax
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  _QWORD *v18; // rax
  const void *v19; // rsi
  __int64 result; // rax
  int v21; // eax
  _QWORD *v22; // rax
  unsigned __int64 v23[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a1 + 24);
  v23[0] = a3;
  sub_20129E0(v5, a2, a3);
  v10 = *(_QWORD *)(a1 + 32);
  if ( (*(_BYTE *)(v10 + 8) & 1) != 0 )
  {
    v11 = v10 + 16;
    v7 = 15;
  }
  else
  {
    v21 = *(_DWORD *)(v10 + 24);
    v11 = *(_QWORD *)(v10 + 16);
    if ( !v21 )
      goto LABEL_15;
    v7 = (unsigned int)(v21 - 1);
  }
  v12 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v13 = (__int64 *)(v11 + 8LL * v12);
  v14 = *v13;
  if ( a2 == *v13 )
  {
LABEL_4:
    *v13 = -16;
    v15 = *(_DWORD *)(v10 + 8);
    v16 = *(_QWORD **)(v10 + 144);
    ++*(_DWORD *)(v10 + 12);
    *(_DWORD *)(v10 + 8) = (2 * (v15 >> 1) - 2) | v15 & 1;
    v6 = *(unsigned int *)(v10 + 152);
    v7 = (__int64)&v16[v6];
    v17 = (8 * v6) >> 3;
    if ( (8 * v6) >> 5 )
    {
      v18 = &v16[4 * ((8 * v6) >> 5)];
      while ( a2 != *v16 )
      {
        if ( a2 == v16[1] )
        {
          v19 = ++v16 + 1;
          goto LABEL_12;
        }
        if ( a2 == v16[2] )
        {
          v16 += 2;
          v19 = v16 + 1;
          goto LABEL_12;
        }
        if ( a2 == v16[3] )
        {
          v16 += 3;
          goto LABEL_11;
        }
        v16 += 4;
        if ( v16 == v18 )
        {
          v17 = (v7 - (__int64)v16) >> 3;
          goto LABEL_25;
        }
      }
      goto LABEL_11;
    }
LABEL_25:
    switch ( v17 )
    {
      case 2LL:
        v22 = v16;
        break;
      case 3LL:
        v19 = v16 + 1;
        v22 = v16 + 1;
        if ( a2 == *v16 )
          goto LABEL_12;
        break;
      case 1LL:
        goto LABEL_32;
      default:
        goto LABEL_28;
    }
    v16 = v22 + 1;
    if ( a2 == *v22 )
    {
      v16 = v22;
      v19 = v22 + 1;
      goto LABEL_12;
    }
LABEL_32:
    if ( a2 == *v16 )
    {
LABEL_11:
      v19 = v16 + 1;
LABEL_12:
      if ( v19 != (const void *)v7 )
      {
        memmove(v16, v19, v7 - (_QWORD)v19);
        v6 = *(unsigned int *)(v10 + 152);
      }
      *(_DWORD *)(v10 + 152) = v6 - 1;
      goto LABEL_15;
    }
LABEL_28:
    v16 = (_QWORD *)v7;
    v19 = (const void *)(v7 + 8);
    goto LABEL_12;
  }
  v6 = 1;
  while ( v14 != -8 )
  {
    v8 = (unsigned int)(v6 + 1);
    v12 = v7 & (v6 + v12);
    v13 = (__int64 *)(v11 + 8LL * v12);
    v14 = *v13;
    if ( *v13 == a2 )
      goto LABEL_4;
    v6 = (unsigned int)v8;
  }
LABEL_15:
  result = v23[0];
  if ( *(_DWORD *)(v23[0] + 28) == -1 )
    return sub_2018710(*(_QWORD *)(a1 + 32), v23, v6, v7, (_QWORD *)v8, v9);
  return result;
}
