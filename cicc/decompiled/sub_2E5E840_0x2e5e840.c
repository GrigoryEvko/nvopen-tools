// Function: sub_2E5E840
// Address: 0x2e5e840
//
void *__fastcall sub_2E5E840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *result; // rax
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  char *v11; // r15
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // r11
  _QWORD *v14; // r10
  int v15; // eax
  _QWORD *v16; // rdi
  _QWORD *v17; // rsi
  bool v18; // al
  __int64 *v19; // rsi
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rax
  size_t v23; // r13
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 *v28; // rdi
  unsigned int v29; // edx
  __int64 *v30; // rsi
  __int64 v31; // rax
  int v32; // esi
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rdx
  const void *v38; // rsi
  _DWORD *v39; // rsi
  __int64 v40; // [rsp+0h] [rbp-70h]
  const void *v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+20h] [rbp-50h]
  int v44; // [rsp+24h] [rbp-4Ch]
  __int64 v45; // [rsp+28h] [rbp-48h]
  __int64 v46; // [rsp+30h] [rbp-40h] BYREF
  __int64 v47[7]; // [rsp+38h] [rbp-38h] BYREF

  result = (void *)(a1 + 176);
  v9 = *(unsigned int *)(a1 + 184);
  v40 = a1 + 176;
  if ( (_DWORD)v9 )
  {
    if ( (void *)a2 != result )
    {
      v33 = *(unsigned int *)(a2 + 8);
      v34 = (unsigned int)v9;
      if ( (unsigned int)v9 <= v33 )
      {
        result = memmove(*(void **)a2, *(const void **)(a1 + 176), 8LL * (unsigned int)v9);
        *(_DWORD *)(a2 + 8) = v9;
      }
      else
      {
        if ( (unsigned int)v9 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
        {
          v39 = (_DWORD *)(a2 + 16);
          v35 = 0;
          *(v39 - 2) = 0;
          sub_C8D5F0(a2, v39, (unsigned int)v9, 8u, a5, a6);
          v34 = *(unsigned int *)(a1 + 184);
        }
        else
        {
          v35 = 8 * v33;
          if ( *(_DWORD *)(a2 + 8) )
          {
            memmove(*(void **)a2, *(const void **)(a1 + 176), 8 * v33);
            v34 = *(unsigned int *)(a1 + 184);
          }
        }
        v36 = *(_QWORD *)(a1 + 176);
        v37 = 8 * v34;
        v38 = (const void *)(v36 + v35);
        result = (void *)(v37 + v36);
        if ( v38 != result )
          result = memcpy((void *)(v35 + *(_QWORD *)a2), v38, v37 - v35);
        *(_DWORD *)(a2 + 8) = v9;
      }
    }
    return result;
  }
  *(_DWORD *)(a2 + 8) = 0;
  v10 = *(_QWORD *)(a1 + 88);
  v11 = *(char **)a2;
  v42 = v10 + 8LL * *(unsigned int *)(a1 + 96);
  if ( v42 == v10 )
  {
    v22 = *(unsigned int *)(a1 + 184);
    v24 = 0;
    if ( *(_DWORD *)(a1 + 188) >= (unsigned int)v22 )
      goto LABEL_26;
    v25 = *(unsigned int *)(a1 + 184);
    v24 = 0;
    v23 = 0;
    goto LABEL_41;
  }
  v45 = *(_QWORD *)(a1 + 88);
  v12 = 0;
  v41 = (const void *)(a2 + 16);
  do
  {
    sub_2E5D970(
      a2,
      &v11[8 * v9],
      *(char **)(*(_QWORD *)v45 + 112LL),
      (char *)(*(_QWORD *)(*(_QWORD *)v45 + 112LL) + 8LL * *(unsigned int *)(*(_QWORD *)v45 + 120LL)));
    v9 = *(unsigned int *)(a2 + 8);
    v13 = v9;
    if ( v9 <= v12 )
    {
      if ( v9 == v12 )
      {
        v11 = *(char **)a2;
      }
      else
      {
LABEL_14:
        if ( *(unsigned int *)(a2 + 12) < v12 )
          sub_C8D5F0(a2, v41, v12, 8u, a5, a6);
        v11 = *(char **)a2;
        v20 = *(_QWORD *)a2 + 8 * v12;
        v21 = (_QWORD *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
        if ( v21 != (_QWORD *)v20 )
        {
          do
          {
            if ( v21 )
              *v21 = 0;
            ++v21;
          }
          while ( (_QWORD *)v20 != v21 );
          v11 = *(char **)a2;
        }
        *(_DWORD *)(a2 + 8) = v12;
        v9 = (unsigned int)v12;
        v13 = (unsigned int)v12;
      }
      goto LABEL_22;
    }
    a5 = v12;
    do
    {
      while ( 1 )
      {
        v14 = *(_QWORD **)a2;
        v15 = *(_DWORD *)(a1 + 72);
        a6 = *(_QWORD *)(*(_QWORD *)a2 + 8 * a5);
        v46 = a6;
        v47[0] = a6;
        if ( !v15 )
        {
          v16 = *(_QWORD **)(a1 + 88);
          v17 = &v16[*(unsigned int *)(a1 + 96)];
          v18 = v17 != sub_2E5D7F0(v16, (__int64)v17, v47);
          goto LABEL_9;
        }
        v26 = *(unsigned int *)(a1 + 80);
        v27 = *(_QWORD *)(a1 + 64);
        v28 = (__int64 *)(v27 + 8 * v26);
        if ( !(_DWORD)v26 )
          break;
        v44 = v26 - 1;
        v29 = (v26 - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4));
        v30 = (__int64 *)(v27 + 8LL * ((*(_DWORD *)(a1 + 80) - 1) & (((unsigned int)a6 >> 9) ^ ((unsigned int)a6 >> 4))));
        v31 = *v30;
        if ( a6 != *v30 )
        {
          v32 = 1;
          while ( v31 != -4096 )
          {
            v29 = v44 & (v32 + v29);
            v43 = v32 + 1;
            v30 = (__int64 *)(v27 + 8LL * v29);
            v31 = *v30;
            if ( a6 == *v30 )
              goto LABEL_30;
            v32 = v43;
          }
          break;
        }
LABEL_30:
        v18 = v30 != v28;
LABEL_9:
        if ( !v18 )
          break;
LABEL_6:
        if ( v13 == ++a5 )
          goto LABEL_12;
      }
      v19 = &v14[v12];
      if ( v19 != sub_2E5D8B0(v14, (__int64)v19, &v46) )
        goto LABEL_6;
      ++a5;
      *v19 = a6;
      ++v12;
    }
    while ( v13 != a5 );
LABEL_12:
    v9 = *(unsigned int *)(a2 + 8);
    v13 = v9;
    if ( v12 != v9 )
    {
      if ( v12 >= v9 )
        goto LABEL_14;
      *(_DWORD *)(a2 + 8) = v12;
      v9 = (unsigned int)v12;
      v13 = (unsigned int)v12;
    }
    v11 = *(char **)a2;
LABEL_22:
    v45 += 8;
  }
  while ( v42 != v45 );
  v22 = *(unsigned int *)(a1 + 184);
  v23 = 8 * v13;
  v24 = v13;
  v25 = v22 + v13;
  if ( v22 + v13 <= *(unsigned int *)(a1 + 188) )
  {
    if ( v23 )
      goto LABEL_25;
    goto LABEL_26;
  }
LABEL_41:
  sub_C8D5F0(v40, (const void *)(a1 + 192), v25, 8u, a5, a6);
  v22 = *(unsigned int *)(a1 + 184);
  if ( v23 )
  {
LABEL_25:
    memcpy((void *)(*(_QWORD *)(a1 + 176) + 8 * v22), v11, v23);
    v22 = *(unsigned int *)(a1 + 184);
  }
LABEL_26:
  result = (void *)(v24 + v22);
  *(_DWORD *)(a1 + 184) = (_DWORD)result;
  return result;
}
