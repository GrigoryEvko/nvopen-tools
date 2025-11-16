// Function: sub_23FB940
// Address: 0x23fb940
//
char *__fastcall sub_23FB940(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4, char *a5, __int64 a6)
{
  _QWORD *v8; // rsi
  char *v9; // r11
  _QWORD *v10; // r9
  int v11; // ecx
  unsigned int v12; // eax
  __int64 v13; // r12
  __int64 v14; // r10
  __int64 v15; // rbx
  __int64 v16; // rdi
  int v17; // eax
  __int64 v19; // r14
  char *v20; // r12
  char *v21; // r15
  __int64 v22; // rcx
  __int64 v23; // rdx
  char *v24; // rdi
  int v25; // r11d
  __int64 v26; // r14
  int v27; // ebx
  __int64 v28; // rsi
  unsigned int v29; // eax
  __int64 v30; // r10
  int v31; // r14d
  size_t v32; // rdx
  int v34; // [rsp-40h] [rbp-40h]

  if ( a4 == 1 )
    return (char *)a1;
  if ( a4 > a6 )
  {
    v19 = a4 / 2;
    v20 = (char *)&a1[a4 / 2];
    v21 = (char *)sub_23FB940(a1, v20, a3, a4 / 2);
    v22 = a4 - v19;
    if ( a4 != v19 )
    {
      v23 = a3;
      v24 = v20;
      v25 = *(_DWORD *)(a3 + 24);
      v26 = *(_QWORD *)(a3 + 8);
      v27 = v25 - 1;
      while ( 1 )
      {
        v28 = *(_QWORD *)(***(_QWORD ***)v24 + 8LL);
        if ( v25 )
        {
          v29 = v27 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
          v30 = *(_QWORD *)(v26 + 8LL * v29);
          if ( v28 == v30 )
          {
LABEL_17:
            v24 = (char *)sub_23FB940(v24, a2, v23, v22);
            return sub_23FB780(v21, v20, v24);
          }
          v34 = 1;
          while ( v30 != -4096 )
          {
            v29 = v27 & (v34 + v29);
            ++v34;
            v30 = *(_QWORD *)(v26 + 8LL * v29);
            if ( v28 == v30 )
              goto LABEL_17;
          }
        }
        v24 += 8;
        if ( !--v22 )
          return sub_23FB780(v21, v20, v24);
      }
    }
    v24 = v20;
    return sub_23FB780(v21, v20, v24);
  }
  v8 = a1 + 1;
  v9 = a5 + 8;
  v10 = a1;
  *(_QWORD *)a5 = *a1;
  if ( a2 == a1 + 1 )
  {
    v32 = 8;
    goto LABEL_10;
  }
  do
  {
    while ( 1 )
    {
      v14 = *v8;
      v15 = *(_QWORD *)(a3 + 8);
      v16 = *(_QWORD *)(**(_QWORD **)*v8 + 8LL);
      v17 = *(_DWORD *)(a3 + 24);
      if ( !v17 )
        goto LABEL_8;
      v11 = v17 - 1;
      v12 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v13 = *(_QWORD *)(v15 + 8LL * v12);
      if ( v16 != v13 )
        break;
LABEL_6:
      ++v8;
      *(_QWORD *)v9 = v14;
      v9 += 8;
      if ( a2 == v8 )
        goto LABEL_9;
    }
    v31 = 1;
    while ( v13 != -4096 )
    {
      v12 = v11 & (v31 + v12);
      v13 = *(_QWORD *)(v15 + 8LL * v12);
      if ( v16 == v13 )
        goto LABEL_6;
      ++v31;
    }
LABEL_8:
    ++v8;
    *v10++ = v14;
  }
  while ( a2 != v8 );
LABEL_9:
  v32 = v9 - a5;
LABEL_10:
  if ( a5 != v9 )
    return (char *)memmove(v10, a5, v32);
  return (char *)v10;
}
