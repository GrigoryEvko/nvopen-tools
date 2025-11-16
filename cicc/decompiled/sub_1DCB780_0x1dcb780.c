// Function: sub_1DCB780
// Address: 0x1dcb780
//
__int64 __fastcall sub_1DCB780(__int64 a1, unsigned int *a2, __int64 a3, __int64 a4, int a5)
{
  unsigned __int64 v7; // r9
  unsigned int *v8; // rdi
  unsigned int *v9; // rcx
  unsigned int v10; // esi
  unsigned int *v11; // rax
  __int64 v13; // r12
  __int64 v14; // r13
  unsigned int v15; // ecx
  unsigned int v16; // edx
  __int64 v17; // rax
  _BOOL4 v18; // r15d
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r13
  unsigned int *v23; // r9
  unsigned int v24; // r15d
  __int64 i; // r12
  unsigned int v26; // edx
  __int64 v27; // rax
  _BOOL4 v28; // r8d
  __int64 v29; // rax
  int v30; // eax
  unsigned int v31; // edx
  __int64 v32; // rax
  int v33; // eax
  unsigned int v34; // ecx
  unsigned int v35; // edx
  __int64 v36; // rax
  _BOOL4 v37; // r12d
  __int64 v38; // rax
  __int64 v39; // rax
  _BOOL4 v40; // [rsp+0h] [rbp-40h]
  __int64 v41; // [rsp+0h] [rbp-40h]
  unsigned int *v42; // [rsp+8h] [rbp-38h]
  unsigned int *v43; // [rsp+8h] [rbp-38h]
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 88) )
  {
    v13 = *(_QWORD *)(a1 + 64);
    v14 = a1 + 56;
    if ( v13 )
    {
      v15 = *a2;
      while ( 1 )
      {
        v16 = *(_DWORD *)(v13 + 32);
        v17 = *(_QWORD *)(v13 + 24);
        if ( v15 < v16 )
          v17 = *(_QWORD *)(v13 + 16);
        if ( !v17 )
          break;
        v13 = v17;
      }
      if ( v15 >= v16 )
      {
        if ( v15 > v16 )
          goto LABEL_16;
LABEL_38:
        v20 = 0;
        return (v20 << 32) | 1;
      }
      if ( *(_QWORD *)(a1 + 72) == v13 )
      {
LABEL_16:
        v18 = 1;
        if ( v14 != v13 )
          v18 = *a2 < *(_DWORD *)(v13 + 32);
        goto LABEL_18;
      }
    }
    else
    {
      v13 = a1 + 56;
      if ( *(_QWORD *)(a1 + 72) == v14 )
      {
        v18 = 1;
LABEL_18:
        v19 = sub_22077B0(40);
        *(_DWORD *)(v19 + 32) = *a2;
        sub_220F040(v18, v19, v13, v14);
        ++*(_QWORD *)(a1 + 88);
        v20 = 1;
        return (v20 << 32) | 1;
      }
    }
    if ( *(_DWORD *)(sub_220EF80(v13) + 32) >= *a2 )
      goto LABEL_38;
    goto LABEL_16;
  }
  v7 = *(unsigned int *)(a1 + 8);
  v8 = *(unsigned int **)a1;
  v9 = &v8[v7];
  if ( v8 != v9 )
  {
    v10 = *a2;
    v11 = v8;
    while ( *v11 != v10 )
    {
      if ( v9 == ++v11 )
        goto LABEL_20;
    }
    if ( v9 != v11 )
      return 1;
  }
LABEL_20:
  if ( v7 > 7 )
  {
    v21 = *(_QWORD *)(a1 + 64);
    v22 = a1 + 56;
    v23 = &v8[v7 - 1];
    if ( v21 )
      goto LABEL_22;
LABEL_34:
    i = a1 + 56;
    if ( *(_QWORD *)(a1 + 72) == v22 )
    {
      v28 = 1;
    }
    else
    {
      v24 = *v23;
LABEL_43:
      v41 = v21;
      v43 = v23;
      v32 = sub_220EF80(i);
      v23 = v43;
      v21 = v41;
      if ( v24 <= *(_DWORD *)(v32 + 32) )
      {
        v33 = *(_DWORD *)(a1 + 8);
        v31 = v33 - 1;
        *(_DWORD *)(a1 + 8) = v33 - 1;
        if ( v33 != 1 )
          goto LABEL_33;
        goto LABEL_45;
      }
LABEL_29:
      v28 = 1;
      if ( i != v22 )
        v28 = v24 < *(_DWORD *)(i + 32);
    }
    v40 = v28;
    v42 = v23;
    v29 = sub_22077B0(40);
    *(_DWORD *)(v29 + 32) = *v42;
    sub_220F040(v40, v29, i, a1 + 56);
    ++*(_QWORD *)(a1 + 88);
    v21 = *(_QWORD *)(a1 + 64);
    while ( 1 )
    {
      v30 = *(_DWORD *)(a1 + 8);
      v31 = v30 - 1;
      *(_DWORD *)(a1 + 8) = v30 - 1;
      if ( v30 == 1 )
        break;
LABEL_33:
      v23 = (unsigned int *)(*(_QWORD *)a1 + 4LL * v31 - 4);
      if ( !v21 )
        goto LABEL_34;
LABEL_22:
      v24 = *v23;
      for ( i = v21; ; i = v27 )
      {
        v26 = *(_DWORD *)(i + 32);
        v27 = *(_QWORD *)(i + 24);
        if ( v24 < v26 )
          v27 = *(_QWORD *)(i + 16);
        if ( !v27 )
          break;
      }
      if ( v24 < v26 )
      {
        if ( *(_QWORD *)(a1 + 72) == i )
          goto LABEL_29;
        goto LABEL_43;
      }
      if ( v24 > v26 )
        goto LABEL_29;
    }
LABEL_45:
    if ( v21 )
    {
      v34 = *a2;
      while ( 1 )
      {
        v35 = *(_DWORD *)(v21 + 32);
        v36 = *(_QWORD *)(v21 + 24);
        if ( v34 < v35 )
          v36 = *(_QWORD *)(v21 + 16);
        if ( !v36 )
          break;
        v21 = v36;
      }
      if ( v34 >= v35 )
      {
        if ( v34 <= v35 )
          return 0x100000001LL;
        goto LABEL_53;
      }
      if ( *(_QWORD *)(a1 + 72) == v21 )
      {
LABEL_53:
        v37 = 1;
        if ( v21 != v22 )
          v37 = *a2 < *(_DWORD *)(v21 + 32);
        goto LABEL_55;
      }
    }
    else
    {
      v21 = a1 + 56;
      if ( *(_QWORD *)(a1 + 72) == v22 )
      {
        v37 = 1;
LABEL_55:
        v44 = v21;
        v38 = sub_22077B0(40);
        *(_DWORD *)(v38 + 32) = *a2;
        sub_220F040(v37, v38, v44, a1 + 56);
        ++*(_QWORD *)(a1 + 88);
        return 0x100000001LL;
      }
    }
    v45 = v21;
    v39 = sub_220EF80(v21);
    v21 = v45;
    if ( *(_DWORD *)(v39 + 32) >= *a2 )
      return 0x100000001LL;
    goto LABEL_53;
  }
  if ( *(_DWORD *)(a1 + 8) >= *(_DWORD *)(a1 + 12) )
  {
    sub_16CD150(a1, (const void *)(a1 + 16), 0, 4, a5, v7);
    v9 = (unsigned int *)(*(_QWORD *)a1 + 4LL * *(unsigned int *)(a1 + 8));
  }
  *v9 = *a2;
  ++*(_DWORD *)(a1 + 8);
  return 0x100000001LL;
}
