// Function: sub_20DDD10
// Address: 0x20ddd10
//
__int64 __fastcall sub_20DDD10(__int64 *a1, __int64 a2, __int64 *a3, unsigned int a4)
{
  __int64 v5; // rax
  void *v6; // rdi
  __int64 v7; // rsi
  unsigned int v8; // r14d
  int v9; // r12d
  unsigned int v10; // eax
  __int64 **v11; // rdx
  __int64 *v12; // rcx
  __int64 *v13; // r8
  char *v14; // rsi
  char *v15; // rdi
  unsigned int v16; // r9d
  __int64 v17; // rsi
  __int64 *v18; // r15
  __int64 v19; // rsi
  __int64 v20; // rax
  char *v21; // rdi
  __int64 *v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // r12
  char *v26; // rdx
  char *v27; // rdi
  const void *v28; // rsi
  bool v30; // al
  bool v31; // al
  unsigned int v32; // [rsp+8h] [rbp-58h]
  __int64 *v35; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v36[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = a1[1];
  v6 = (void *)*a1;
  v35 = a3;
  v7 = v5 - (_QWORD)v6;
  if ( v5 - (__int64)v6 > 16 )
  {
    qsort(v6, v7 >> 4, 0x10u, (__compar_fn_t)sub_20D63A0);
    v5 = a1[1];
    v7 = v5 - *a1;
  }
  v8 = 0;
  if ( (unsigned __int64)v7 <= 0x10 )
    return v8;
  do
  {
    v9 = *(_DWORD *)(v5 - 16);
    v10 = sub_20DBE00((__int64)a1, v9, a4, a2, v35);
    v14 = (char *)a1[15];
    v15 = (char *)a1[14];
    v16 = v10;
    if ( v15 == v14 )
      goto LABEL_31;
    v17 = v14 - v15;
    v18 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*a1 + 8) + 56LL) + 328LL);
    v36[0] = v17 >> 4;
    if ( v17 == 32 )
    {
      v32 = v10;
      v30 = sub_1DD69A0(*(_QWORD *)(*(_QWORD *)v15 + 8LL), *(_QWORD *)(*((_QWORD *)v15 + 2) + 8LL));
      v15 = (char *)a1[14];
      v16 = v32;
      v17 = a1[15] - (_QWORD)v15;
      if ( v30 )
      {
        v12 = (__int64 *)*((_QWORD *)v15 + 3);
        if ( *(__int64 **)(*(_QWORD *)(*((_QWORD *)v15 + 2) + 8LL) + 32LL) == v12 )
        {
          v36[0] = 1;
          v19 = v17 >> 4;
          v20 = 1;
          goto LABEL_16;
        }
      }
      if ( v17 == 32 )
      {
        v31 = sub_1DD69A0(*(_QWORD *)(*((_QWORD *)v15 + 2) + 8LL), *(_QWORD *)(*(_QWORD *)v15 + 8LL));
        v15 = (char *)a1[14];
        v16 = v32;
        v17 = a1[15] - (_QWORD)v15;
        if ( v31 )
        {
          v12 = (__int64 *)*((_QWORD *)v15 + 1);
          if ( *(__int64 **)(*(_QWORD *)(*(_QWORD *)v15 + 8LL) + 32LL) == v12 )
          {
            v36[0] = 0;
            v19 = v17 >> 4;
            v20 = 0;
            goto LABEL_16;
          }
        }
      }
    }
    v19 = v17 >> 4;
    if ( !(_DWORD)v19 )
    {
LABEL_30:
      v20 = v36[0];
      goto LABEL_16;
    }
    v11 = (__int64 **)(v15 + 8);
    v20 = 0;
    while ( 1 )
    {
      LODWORD(v13) = v20;
      v12 = (__int64 *)(*(v11 - 1))[1];
      if ( v18 == v12 )
        break;
      if ( v35 == v12 )
        goto LABEL_15;
      v13 = *v11;
      if ( (__int64 *)v12[4] == *v11 )
        v36[0] = v20;
LABEL_11:
      ++v20;
      v11 += 2;
      if ( (unsigned int)v19 == v20 )
        goto LABEL_30;
    }
    v12 = *v11;
    if ( (__int64 *)v18[4] == *v11 || v35 != v18 )
      goto LABEL_11;
LABEL_15:
    v36[0] = v20;
LABEL_16:
    if ( v19 == v20
      || (v21 = &v15[16 * v20], v22 = *(__int64 **)(*(_QWORD *)v21 + 8LL), v35 == v22) && v22[4] != *((_QWORD *)v21 + 1) )
    {
      if ( (unsigned __int8)sub_20DD5D0((__int64)a1, (__int64 *)&v35, a2, v16, v36) )
      {
        v22 = *(__int64 **)(*(_QWORD *)(a1[14] + 16LL * v36[0]) + 8LL);
        goto LABEL_21;
      }
LABEL_31:
      sub_20D7530(a1, v9, a2, (__int64)v35);
    }
    else
    {
LABEL_21:
      v23 = 0;
      sub_20DD8B0(a1, (__int64)v22, (__int64)v11, (__int64)v12, (int)v13, v16);
      sub_20D75F0((__int64)a1, v36[0]);
      v24 = (a1[15] - a1[14]) >> 4;
      v25 = (unsigned int)v24;
      if ( (_DWORD)v24 )
      {
        do
        {
          if ( v36[0] != (_DWORD)v23 )
          {
            sub_20D7260((__int64)a1, *(_QWORD **)(a1[14] + 16 * v23 + 8), (__int64)v22);
            v26 = (char *)a1[1];
            v27 = *(char **)(a1[14] + 16 * v23);
            v28 = v27 + 16;
            if ( v26 != v27 + 16 )
            {
              memmove(v27, v28, v26 - (_BYTE *)v28);
              v28 = (const void *)a1[1];
            }
            a1[1] = (__int64)v28 - 16;
          }
          ++v23;
        }
        while ( v25 != v23 );
      }
      v8 = 1;
    }
    v5 = a1[1];
  }
  while ( (unsigned __int64)(v5 - *a1) > 0x10 );
  return v8;
}
