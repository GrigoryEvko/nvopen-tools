// Function: sub_C8D5F0
// Address: 0xc8d5f0
//
__int64 __fastcall sub_C8D5F0(
        __int64 a1,
        const void *a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  void *v12; // rdi
  __int64 v13; // r15
  __int64 v14; // rsi
  __int64 result; // rax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  void *v20; // r13
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  void *v30; // r14
  void *v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // [rsp+0h] [rbp-40h]
  void *v39; // [rsp+0h] [rbp-40h]

  v6 = 0xFFFFFFFFLL;
  v7 = a3;
  v9 = *(unsigned int *)(a1 + 12);
  if ( a3 > 0xFFFFFFFF )
    sub_C8D440(a3, (__int64)a2, a3, a4);
  if ( v9 == 0xFFFFFFFFLL )
    sub_C8D230(0xFFFFFFFF, (__int64)a2, a3, a4);
  v10 = 2 * v9 + 1;
  if ( v7 > v10 )
  {
    v12 = *(void **)a1;
    v13 = v7 * a4;
    if ( *(const void **)a1 != a2 )
      goto LABEL_5;
LABEL_12:
    v20 = (void *)malloc(v13, a2, v10, a4, a5, a6);
    if ( v20 || !v13 && (v20 = (void *)malloc(1, a2, v21, v22, v23, v24)) != 0 )
    {
      if ( a2 == v20 )
      {
        v32 = malloc(v13, a2, v21, v22, v23, v24);
        if ( !v32 )
        {
          if ( v13 )
            goto LABEL_25;
          v32 = malloc(1, a2, v33, v34, v35, v36);
          if ( !v32 )
            goto LABEL_25;
        }
        v39 = (void *)v32;
        _libc_free(v20, a2);
        v20 = v39;
      }
      result = (__int64)memcpy(v20, a2, a4 * *(unsigned int *)(a1 + 8));
      goto LABEL_8;
    }
LABEL_25:
    sub_C64F00("Allocation failed", 1u);
  }
  v12 = *(void **)a1;
  if ( v10 <= 0xFFFFFFFF )
    v6 = v10;
  LODWORD(v7) = v6;
  v13 = v6 * a4;
  if ( v12 == a2 )
    goto LABEL_12;
LABEL_5:
  v14 = v13;
  result = realloc(v12);
  v20 = (void *)result;
  if ( !result )
  {
    v14 = v13;
    result = sub_C8D200((__int64)v12, v13, v16, v17, v18, v19);
    v20 = (void *)result;
  }
  if ( a2 == v20 )
  {
    v38 = *(unsigned int *)(a1 + 8);
    v25 = malloc(v13, v14, v38, v17, v18, v19);
    v29 = v38;
    v30 = (void *)v25;
    if ( v25 || !v13 && (v37 = malloc(1, v14, v38, v26, v27, v28), v29 = v38, (v30 = (void *)v37) != 0) )
    {
      if ( v29 )
      {
        v14 = (__int64)v20;
        memcpy(v30, v20, a4 * v29);
      }
      v31 = v20;
      v20 = v30;
      result = _libc_free(v31, v14);
      goto LABEL_8;
    }
    goto LABEL_25;
  }
LABEL_8:
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 12) = v7;
  return result;
}
