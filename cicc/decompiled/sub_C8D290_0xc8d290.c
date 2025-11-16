// Function: sub_C8D290
// Address: 0xc8d290
//
__int64 __fastcall sub_C8D290(__int64 a1, const void *a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // rdx
  void *v10; // rdi
  __int64 v12; // r14
  __int64 v13; // rsi
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  void *v19; // r12
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  void *v29; // r15
  void *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // [rsp+0h] [rbp-40h]
  void *v38; // [rsp+0h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 16);
  if ( v6 == -1 )
    sub_C8D230(0xFFFFFFFFFFFFFFFFLL, (__int64)a2, a3, a4);
  v7 = a3;
  v8 = 2 * v6 + 1;
  v10 = *(void **)a1;
  if ( v8 >= v7 )
    v7 = 2 * v6 + 1;
  v12 = v7 * a4;
  if ( v10 == a2 )
  {
    v19 = (void *)malloc(v12, a2, v8, a4, a5, a6);
    if ( v19 || !v12 && (v19 = (void *)malloc(1, a2, v20, v21, v22, v23)) != 0 )
    {
      if ( a2 == v19 )
      {
        v31 = malloc(v12, a2, v20, v21, v22, v23);
        if ( !v31 )
        {
          if ( v12 )
            goto LABEL_21;
          v31 = malloc(1, a2, v32, v33, v34, v35);
          if ( !v31 )
            goto LABEL_21;
        }
        v38 = (void *)v31;
        _libc_free(v19, a2);
        v19 = v38;
      }
      result = (__int64)memcpy(v19, a2, *(_QWORD *)(a1 + 8) * a4);
      goto LABEL_8;
    }
LABEL_21:
    sub_C64F00("Allocation failed", 1u);
  }
  v13 = v7 * a4;
  result = realloc(v10);
  v19 = (void *)result;
  if ( !result )
  {
    v13 = v12;
    result = sub_C8D200((__int64)v10, v12, v15, v16, v17, v18);
    v19 = (void *)result;
  }
  if ( a2 == v19 )
  {
    v37 = *(_QWORD *)(a1 + 8);
    v24 = malloc(v12, v13, v37, v16, v17, v18);
    v28 = v37;
    v29 = (void *)v24;
    if ( v24 || !v12 && (v36 = malloc(1, v13, v37, v25, v26, v27), v28 = v37, (v29 = (void *)v36) != 0) )
    {
      if ( v28 )
      {
        v13 = (__int64)v19;
        memcpy(v29, v19, a4 * v28);
      }
      v30 = v19;
      v19 = v29;
      result = _libc_free(v30, v13);
      goto LABEL_8;
    }
    goto LABEL_21;
  }
LABEL_8:
  *(_QWORD *)a1 = v19;
  *(_QWORD *)(a1 + 16) = v7;
  return result;
}
