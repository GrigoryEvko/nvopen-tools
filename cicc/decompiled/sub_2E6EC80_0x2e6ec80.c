// Function: sub_2E6EC80
// Address: 0x2e6ec80
//
_QWORD *__fastcall sub_2E6EC80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 v8; // r9
  char *v9; // rsi
  __int64 v10; // r13
  __int64 v11; // r9
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdi
  int v21; // esi
  __int64 *v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 *v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // rsi
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 v31; // [rsp+0h] [rbp-40h]
  __int64 v32; // [rsp+8h] [rbp-38h]
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  v7 = *(unsigned int *)(a2 + 120);
  v8 = *(_QWORD *)(a2 + 112);
  v9 = (char *)(a1 + 2);
  v10 = 8 * v7;
  v11 = v10 + v8;
  v12 = v10 >> 3;
  v13 = v10 >> 3;
  if ( a3 )
  {
    v14 = *(_QWORD *)(a3 + 8);
    *a1 = v9;
    a1[1] = 0x800000000LL;
    LODWORD(v15) = 0;
    if ( (unsigned __int64)v10 > 0x40 )
    {
      v31 = v11;
      v33 = v14;
      sub_C8D5F0((__int64)a1, v9, v10 >> 3, 8u, v14, v11);
      v15 = *((unsigned int *)a1 + 2);
      v11 = v31;
      v14 = v33;
      v9 = (char *)(*a1 + 8 * v15);
    }
    if ( v10 )
    {
      do
      {
        v9 += 8;
        *((_QWORD *)v9 - 1) = *(_QWORD *)(v11 - 8 * v12 + 8 * v13-- - 8);
      }
      while ( v13 );
      LODWORD(v15) = *((_DWORD *)a1 + 2);
    }
    v32 = v14;
    *((_DWORD *)a1 + 2) = v12 + v15;
    sub_2E6E9F0((__int64)a1);
    v17 = v32;
    v18 = *(_BYTE *)(v32 + 8) & 1;
    if ( (*(_BYTE *)(v32 + 8) & 1) != 0 )
    {
      v20 = v32 + 16;
      v21 = 3;
    }
    else
    {
      v19 = *(unsigned int *)(v32 + 24);
      v20 = *(_QWORD *)(v32 + 16);
      if ( !(_DWORD)v19 )
        goto LABEL_26;
      v21 = v19 - 1;
    }
    v19 = v21 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v22 = (__int64 *)(v20 + 72 * v19);
    v23 = *v22;
    if ( a2 == *v22 )
    {
LABEL_11:
      v24 = 288;
      if ( !(_BYTE)v18 )
        v24 = 72LL * *(unsigned int *)(v32 + 24);
      if ( v22 != (__int64 *)(v20 + v24) )
      {
        v25 = (__int64 *)v22[1];
        v26 = &v25[*((unsigned int *)v22 + 4)];
        while ( v26 != v25 )
        {
          v27 = *v25++;
          sub_2E6EB60((__int64)a1, v27);
        }
        sub_2E6EC00((__int64)a1, (__int64)(v22 + 5), v19, v18, v17, v16);
      }
      return a1;
    }
    v16 = 1;
    while ( v23 != -4096 )
    {
      v19 = v21 & (unsigned int)(v16 + v19);
      v22 = (__int64 *)(v20 + 72LL * (unsigned int)v19);
      v23 = *v22;
      if ( a2 == *v22 )
        goto LABEL_11;
      v16 = (unsigned int)(v16 + 1);
    }
    if ( (_BYTE)v18 )
    {
      v30 = 288;
      goto LABEL_27;
    }
    v19 = *(unsigned int *)(v32 + 24);
LABEL_26:
    v30 = 72 * v19;
LABEL_27:
    v22 = (__int64 *)(v20 + v30);
    goto LABEL_11;
  }
  *a1 = v9;
  a1[1] = 0x800000000LL;
  LODWORD(v29) = 0;
  if ( (unsigned __int64)v10 > 0x40 )
  {
    v34 = v11;
    sub_C8D5F0((__int64)a1, v9, v10 >> 3, 8u, a5, v11);
    v29 = *((unsigned int *)a1 + 2);
    v11 = v34;
    v9 = (char *)(*a1 + 8 * v29);
  }
  if ( v10 )
  {
    do
    {
      v9 += 8;
      *((_QWORD *)v9 - 1) = *(_QWORD *)(v11 - 8 * v12 + 8 * v13-- - 8);
    }
    while ( v13 );
    LODWORD(v29) = *((_DWORD *)a1 + 2);
  }
  *((_DWORD *)a1 + 2) = v12 + v29;
  sub_2E6E9F0((__int64)a1);
  return a1;
}
