// Function: sub_2C6F8B0
// Address: 0x2c6f8b0
//
__int64 *__fastcall sub_2C6F8B0(__int64 **a1, __int64 **a2, __int64 **a3, __int64 **a4)
{
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // rax
  size_t v10; // rdx
  void *v11; // rcx
  int v12; // eax
  const char *v13; // rax
  size_t v14; // rdx
  size_t v15; // rbx
  const char *v16; // rax
  size_t v17; // rdx
  void *v18; // rcx
  int v19; // eax
  __int64 *v20; // rbx
  const char *v21; // rax
  size_t v22; // rdx
  size_t v23; // r12
  const char *v24; // rax
  size_t v25; // rdx
  size_t v26; // rbx
  int v27; // eax
  __int64 *result; // rax
  const char *v29; // rax
  size_t v30; // rdx
  size_t v31; // rbx
  const char *v32; // rax
  size_t v33; // rdx
  void *v34; // rcx
  int v35; // eax
  __int64 *v36; // r13
  const char *v37; // rax
  size_t v38; // rdx
  size_t v39; // rbx
  const char *v40; // rax
  size_t v41; // rdx
  size_t v42; // r13
  bool v43; // cc
  size_t v44; // rdx
  int v45; // eax
  const char *s2; // [rsp+0h] [rbp-40h]
  const char *s2a; // [rsp+0h] [rbp-40h]
  const char *s2b; // [rsp+0h] [rbp-40h]
  __int64 *v49; // [rsp+8h] [rbp-38h]
  void *v50; // [rsp+8h] [rbp-38h]
  __int64 *v51; // [rsp+8h] [rbp-38h]
  void *v52; // [rsp+8h] [rbp-38h]
  const char *v53; // [rsp+8h] [rbp-38h]
  __int64 *v54; // [rsp+8h] [rbp-38h]
  void *v55; // [rsp+8h] [rbp-38h]
  const char *v56; // [rsp+8h] [rbp-38h]

  v49 = *a2;
  v6 = sub_BD5D20(**a3);
  v8 = v7;
  s2 = v6;
  v9 = sub_BD5D20(*v49);
  v11 = (void *)v10;
  if ( v8 <= v10 )
    v10 = v8;
  if ( v10 )
  {
    v50 = v11;
    v12 = memcmp(v9, s2, v10);
    v11 = v50;
    if ( v12 )
    {
      if ( v12 < 0 )
        goto LABEL_7;
LABEL_21:
      v54 = *a2;
      v29 = sub_BD5D20(**a4);
      v31 = v30;
      s2b = v29;
      v32 = sub_BD5D20(*v54);
      v34 = (void *)v33;
      if ( v31 <= v33 )
        v33 = v31;
      if ( v33 && (v55 = v34, v35 = memcmp(v32, s2b, v33), v34 = v55, v35) )
      {
        if ( v35 < 0 )
          goto LABEL_19;
      }
      else if ( (void *)v31 != v34 && v31 > (unsigned __int64)v34 )
      {
        goto LABEL_19;
      }
      v36 = *a3;
      v37 = sub_BD5D20(**a4);
      v39 = v38;
      v56 = v37;
      v40 = sub_BD5D20(*v36);
      v42 = v41;
      v43 = v41 <= v39;
      v44 = v39;
      if ( v43 )
        v44 = v42;
      if ( v44 && (v45 = memcmp(v40, v56, v44)) != 0 )
      {
        if ( v45 < 0 )
          goto LABEL_33;
      }
      else if ( v42 != v39 && v42 < v39 )
      {
        goto LABEL_33;
      }
      goto LABEL_35;
    }
  }
  if ( (void *)v8 == v11 || v8 <= (unsigned __int64)v11 )
    goto LABEL_21;
LABEL_7:
  v51 = *a3;
  v13 = sub_BD5D20(**a4);
  v15 = v14;
  s2a = v13;
  v16 = sub_BD5D20(*v51);
  v18 = (void *)v17;
  if ( v15 <= v17 )
    v17 = v15;
  if ( v17 && (v52 = v18, v19 = memcmp(v16, s2a, v17), v18 = v52, v19) )
  {
    if ( v19 >= 0 )
      goto LABEL_13;
  }
  else if ( (void *)v15 == v18 || v15 <= (unsigned __int64)v18 )
  {
LABEL_13:
    v20 = *a2;
    v21 = sub_BD5D20(**a4);
    v23 = v22;
    v53 = v21;
    v24 = sub_BD5D20(*v20);
    v26 = v25;
    if ( v23 <= v25 )
      v25 = v23;
    if ( !v25 || (v27 = memcmp(v24, v53, v25)) == 0 )
    {
      if ( v23 == v26 || v23 <= v26 )
        goto LABEL_19;
      goto LABEL_33;
    }
    if ( v27 < 0 )
    {
LABEL_33:
      result = *a1;
      *a1 = *a4;
      *a4 = result;
      return result;
    }
LABEL_19:
    result = *a1;
    *a1 = *a2;
    *a2 = result;
    return result;
  }
LABEL_35:
  result = *a1;
  *a1 = *a3;
  *a3 = result;
  return result;
}
