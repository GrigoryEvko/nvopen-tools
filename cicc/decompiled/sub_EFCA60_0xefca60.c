// Function: sub_EFCA60
// Address: 0xefca60
//
__int64 __fastcall sub_EFCA60(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r9
  char *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // r8
  unsigned __int64 v12; // rcx
  __int64 result; // rax
  char *v14; // r10
  unsigned __int64 v15; // rdx
  __int64 v16; // r14
  unsigned __int64 v17; // rcx
  unsigned int v18; // esi
  unsigned __int64 v19; // rcx
  void *v20; // rdi
  __int64 v21; // r9
  signed __int64 v22; // rax
  char *v23; // r8
  __int64 v24; // r11
  unsigned __int64 v25; // rdx
  __int64 v26; // [rsp+8h] [rbp-58h]
  int v27; // [rsp+10h] [rbp-50h]
  signed __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h]
  char *v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  char *v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35; // [rsp+28h] [rbp-38h]
  char *v36; // [rsp+28h] [rbp-38h]
  char *v37; // [rsp+28h] [rbp-38h]
  char *v38; // [rsp+28h] [rbp-38h]

  v5 = a4;
  v7 = a2;
  v8 = a4 - a3;
  v9 = a3;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1;
  v12 = *(unsigned int *)(a1 + 12);
  result = 8 * v10;
  v14 = &a2[-*(_QWORD *)a1];
  v15 = v10 + v8;
  v16 = *(_QWORD *)a1 + 8 * v10;
  if ( v7 == (char *)v16 )
  {
    if ( v15 > v12 )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 8u, v11, v5);
      v10 = *(unsigned int *)(a1 + 8);
      result = *(_QWORD *)a1;
      v16 = *(_QWORD *)a1 + 8 * v10;
    }
    if ( v8 > 0 )
    {
      for ( result = 0; result != v8; ++result )
        *(_QWORD *)(v16 + 8 * result) = *(char *)(v9 + result);
      LODWORD(v10) = *(_DWORD *)(a1 + 8);
    }
    *(_DWORD *)(a1 + 8) = v8 + v10;
  }
  else
  {
    if ( v15 > v12 )
    {
      v32 = v5;
      v36 = v14;
      sub_C8D5F0(a1, (const void *)(a1 + 16), v15, 8u, v11, v5);
      v10 = *(unsigned int *)(a1 + 8);
      v11 = *(_QWORD *)a1;
      v14 = v36;
      v5 = v32;
      result = 8 * v10;
      v7 = &v36[*(_QWORD *)a1];
      v16 = *(_QWORD *)a1 + 8 * v10;
    }
    v17 = (result - (__int64)v14) >> 3;
    if ( v17 >= v8 )
    {
      v19 = *(unsigned int *)(a1 + 12);
      v20 = (void *)v16;
      v21 = 8 * (v10 - v8);
      v22 = result - v21;
      v23 = (char *)(v21 + v11);
      v24 = v22 >> 3;
      v25 = (v22 >> 3) + v10;
      if ( v25 > v19 )
      {
        v26 = v22 >> 3;
        v28 = v22;
        v30 = v23;
        v34 = 8 * (v10 - v8);
        v38 = v14;
        sub_C8D5F0(a1, (const void *)(a1 + 16), v25, 8u, (__int64)v23, v21);
        v10 = *(unsigned int *)(a1 + 8);
        LODWORD(v24) = v26;
        v22 = v28;
        v23 = v30;
        v21 = v34;
        v20 = (void *)(*(_QWORD *)a1 + 8 * v10);
        v14 = v38;
      }
      if ( v23 != (char *)v16 )
      {
        v27 = v24;
        v29 = v21;
        v33 = v14;
        v37 = v23;
        memmove(v20, v23, v22);
        LODWORD(v10) = *(_DWORD *)(a1 + 8);
        LODWORD(v24) = v27;
        v21 = v29;
        v14 = v33;
        v23 = v37;
      }
      *(_DWORD *)(a1 + 8) = v24 + v10;
      if ( v23 != v7 )
        memmove((void *)(v16 - (v21 - (_QWORD)v14)), v7, v21 - (_QWORD)v14);
      result = 0;
      if ( v8 > 0 )
      {
        do
        {
          *(_QWORD *)&v7[8 * result] = *(char *)(v9 + result);
          ++result;
        }
        while ( result != v8 );
      }
    }
    else
    {
      v18 = v8 + v10;
      *(_DWORD *)(a1 + 8) = v18;
      if ( (char *)v16 != v7 )
      {
        v31 = (result - (__int64)v14) >> 3;
        v35 = v5;
        result = (__int64)memcpy((void *)(v11 + 8LL * v18 - (result - (_QWORD)v14)), v7, result - (_QWORD)v14);
        v17 = v31;
        v5 = v35;
      }
      if ( v17 )
      {
        for ( result = 0; result != v17; ++result )
          *(_QWORD *)&v7[8 * result] = *(char *)(v9 + result);
        v9 += v17;
        v8 = v5 - v9;
      }
      if ( v8 > 0 )
      {
        for ( result = 0; result != v8; ++result )
          *(_QWORD *)(v16 + 8 * result) = *(char *)(v9 + result);
      }
    }
  }
  return result;
}
