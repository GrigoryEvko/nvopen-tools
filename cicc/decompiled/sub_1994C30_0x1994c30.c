// Function: sub_1994C30
// Address: 0x1994c30
//
__int64 __fastcall sub_1994C30(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  unsigned __int64 v8; // r8
  __int64 *v9; // rdi
  __int64 *v10; // rcx
  __int64 v11; // rsi
  __int64 result; // rax
  _QWORD *v13; // r12
  _QWORD *v14; // r13
  __int64 v15; // rcx
  __int64 v16; // rdx
  _BOOL4 v17; // r15d
  __int64 v18; // rax
  _QWORD *v19; // r15
  _QWORD *v20; // r13
  __int64 v21; // rcx
  _QWORD *i; // r12
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _BOOL4 v25; // r15d
  __int64 v26; // rax
  int v27; // eax
  unsigned int v28; // edx
  __int64 v29; // rax
  int v30; // eax
  __int64 v31; // r12
  __int64 v32; // rdx
  _QWORD *v33; // rax
  _BOOL4 v34; // r8d
  __int64 v35; // rax
  _QWORD *v36; // [rsp+8h] [rbp-38h]
  _QWORD *v37; // [rsp+8h] [rbp-38h]
  _BOOL4 v38; // [rsp+8h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 120) )
  {
    v13 = *(_QWORD **)(a1 + 96);
    v14 = (_QWORD *)(a1 + 88);
    if ( v13 )
    {
      v15 = *a2;
      while ( 1 )
      {
        v16 = v13[4];
        result = v13[3];
        if ( v15 < v16 )
          result = v13[2];
        if ( !result )
          break;
        v13 = (_QWORD *)result;
      }
      if ( v15 >= v16 )
      {
        if ( v15 <= v16 )
          return result;
        goto LABEL_16;
      }
      if ( v13 == *(_QWORD **)(a1 + 104) )
      {
LABEL_16:
        v17 = 1;
        if ( v14 != v13 )
          v17 = *a2 < v13[4];
        goto LABEL_18;
      }
    }
    else
    {
      v13 = (_QWORD *)(a1 + 88);
      if ( v14 == *(_QWORD **)(a1 + 104) )
      {
        v17 = 1;
LABEL_18:
        v18 = sub_22077B0(40);
        *(_QWORD *)(v18 + 32) = *a2;
        sub_220F040(v17, v18, v13, v14);
        ++*(_QWORD *)(a1 + 120);
        goto LABEL_19;
      }
    }
    result = sub_220EF80(v13);
    if ( *(_QWORD *)(result + 32) >= *a2 )
      return result;
    goto LABEL_16;
  }
  v8 = *(unsigned int *)(a1 + 8);
  v9 = *(__int64 **)a1;
  v10 = &v9[v8];
  if ( v9 == v10 )
    goto LABEL_22;
  v11 = *a2;
  result = (__int64)v9;
  while ( *(_QWORD *)result != v11 )
  {
    result += 8;
    if ( v10 == (__int64 *)result )
      goto LABEL_22;
  }
  if ( v10 == (__int64 *)result )
  {
LABEL_22:
    if ( v8 <= 7 )
    {
      if ( *(_DWORD *)(a1 + 8) >= *(_DWORD *)(a1 + 12) )
      {
        sub_16CD150(a1, (const void *)(a1 + 16), 0, 8, v8, a6);
        v10 = (__int64 *)(*(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8));
      }
      *v10 = *a2;
      ++*(_DWORD *)(a1 + 8);
      goto LABEL_19;
    }
    v19 = *(_QWORD **)(a1 + 96);
    v20 = (_QWORD *)(a1 + 88);
    v8 = (unsigned __int64)&v9[v8 - 1];
    if ( v19 )
      goto LABEL_24;
LABEL_36:
    i = (_QWORD *)(a1 + 88);
    if ( *(_QWORD **)(a1 + 104) == v20 )
    {
      v25 = 1;
    }
    else
    {
      do
      {
        v37 = (_QWORD *)v8;
        v29 = sub_220EF80(i);
        v8 = (unsigned __int64)v37;
        if ( *(_QWORD *)(v29 + 32) < *v37 )
          break;
        v30 = *(_DWORD *)(a1 + 8);
        v28 = v30 - 1;
        *(_DWORD *)(a1 + 8) = v30 - 1;
        if ( v30 == 1 )
          goto LABEL_47;
LABEL_35:
        v8 = *(_QWORD *)a1 + 8LL * v28 - 8;
        if ( !v19 )
          goto LABEL_36;
LABEL_24:
        v21 = *(_QWORD *)v8;
        for ( i = v19; ; i = v24 )
        {
          v23 = i[4];
          v24 = (_QWORD *)i[3];
          if ( v21 < v23 )
            v24 = (_QWORD *)i[2];
          if ( !v24 )
            break;
        }
        if ( v21 >= v23 )
        {
          if ( v21 <= v23 )
            goto LABEL_34;
          break;
        }
      }
      while ( *(_QWORD **)(a1 + 104) != i );
      v25 = 1;
      if ( i != v20 )
        v25 = *(_QWORD *)v8 < i[4];
    }
    v36 = (_QWORD *)v8;
    v26 = sub_22077B0(40);
    *(_QWORD *)(v26 + 32) = *v36;
    sub_220F040(v25, v26, i, a1 + 88);
    ++*(_QWORD *)(a1 + 120);
    v19 = *(_QWORD **)(a1 + 96);
LABEL_34:
    v27 = *(_DWORD *)(a1 + 8);
    v28 = v27 - 1;
    *(_DWORD *)(a1 + 8) = v27 - 1;
    if ( v27 != 1 )
      goto LABEL_35;
LABEL_47:
    if ( v19 )
    {
      v31 = *a2;
      while ( 1 )
      {
        v32 = v19[4];
        v33 = (_QWORD *)v19[3];
        if ( v31 < v32 )
          v33 = (_QWORD *)v19[2];
        if ( !v33 )
          break;
        v19 = v33;
      }
      if ( v31 >= v32 )
      {
        if ( v31 > v32 )
          goto LABEL_55;
LABEL_19:
        result = *(unsigned int *)(a1 + 136);
        if ( (unsigned int)result >= *(_DWORD *)(a1 + 140) )
        {
          sub_16CD150(a1 + 128, (const void *)(a1 + 144), 0, 8, v8, a6);
          result = *(unsigned int *)(a1 + 136);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 128) + 8 * result) = *a2;
        ++*(_DWORD *)(a1 + 136);
        return result;
      }
      if ( *(_QWORD **)(a1 + 104) == v19 )
      {
LABEL_55:
        v34 = 1;
        if ( v19 != v20 )
          v34 = v31 < v19[4];
        goto LABEL_57;
      }
    }
    else
    {
      v19 = (_QWORD *)(a1 + 88);
      if ( *(_QWORD **)(a1 + 104) == v20 )
      {
        v34 = 1;
LABEL_57:
        v38 = v34;
        v35 = sub_22077B0(40);
        *(_QWORD *)(v35 + 32) = *a2;
        sub_220F040(v38, v35, v19, a1 + 88);
        ++*(_QWORD *)(a1 + 120);
        goto LABEL_19;
      }
      v31 = *a2;
    }
    if ( v31 <= *(_QWORD *)(sub_220EF80(v19) + 32) )
      goto LABEL_19;
    goto LABEL_55;
  }
  return result;
}
