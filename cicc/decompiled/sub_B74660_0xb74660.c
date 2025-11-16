// Function: sub_B74660
// Address: 0xb74660
//
__int64 __fastcall sub_B74660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rbx
  int *v11; // r15
  int *v12; // r13
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rcx
  int *v16; // rdi
  _QWORD *v18; // r14
  _QWORD *v19; // rax
  int *v20; // rdi
  char v21; // al
  __int64 v22; // rax
  _QWORD *m; // rbx
  __int64 v24; // rax
  _QWORD *k; // r12
  _QWORD *j; // r15
  _QWORD *i; // r15
  __int64 v28; // [rsp-98h] [rbp-98h] BYREF
  _QWORD *v29; // [rsp-90h] [rbp-90h]
  int v30; // [rsp-78h] [rbp-78h]
  char v31; // [rsp-74h] [rbp-74h]
  __int64 v32; // [rsp-70h] [rbp-70h] BYREF
  _QWORD *v33; // [rsp-68h] [rbp-68h]
  __int64 v34; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v35; // [rsp-50h] [rbp-50h] BYREF
  _QWORD *v36; // [rsp-48h] [rbp-48h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
    return result;
  v6 = sub_C33690(a1, a2, a3, a4, a5);
  v10 = sub_C33340(a1, a2, v7, v8, v9);
  if ( v6 == v10 )
    sub_C3C5A0(&v34, v6, 1);
  else
    sub_C36740(&v34, v6, 1);
  v31 = 1;
  v30 = -1;
  if ( v34 == v10 )
    sub_C3C840(&v32, &v34);
  else
    sub_C338E0(&v32, &v34);
  if ( v34 == v10 )
  {
    if ( v35 )
    {
      for ( i = &v35[3 * *(v35 - 1)]; v35 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v34);
  }
  if ( v6 == v10 )
    sub_C3C5A0(&v28, v10, 2);
  else
    sub_C36740(&v28, v6, 2);
  BYTE4(v34) = 0;
  LODWORD(v34) = -2;
  if ( v10 == v28 )
    sub_C3C840(&v35, &v28);
  else
    sub_C338E0(&v35, &v28);
  if ( v10 == v28 )
  {
    if ( v29 )
    {
      for ( j = &v29[3 * *(v29 - 1)]; v29 != j; sub_91D830(j) )
        j -= 3;
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0(&v28);
  }
  v11 = *(int **)(a1 + 8);
  result = 5LL * *(unsigned int *)(a1 + 24);
  v12 = &v11[10 * *(unsigned int *)(a1 + 24)];
  if ( v11 != v12 )
  {
    while ( 1 )
    {
      v14 = *v11;
      if ( *v11 != v30 )
        goto LABEL_16;
      if ( *((_BYTE *)v11 + 4) != v31 )
        goto LABEL_16;
      v15 = *((_QWORD *)v11 + 1);
      if ( v15 != v32 )
        goto LABEL_16;
      v16 = v11 + 2;
      if ( !(v10 == v15 ? sub_C3E590(v16) : (unsigned __int8)sub_C33D00(v16)) )
        break;
LABEL_19:
      if ( v10 == *((_QWORD *)v11 + 1) )
      {
        result = *((_QWORD *)v11 + 2);
        if ( !result )
          goto LABEL_21;
        v18 = (_QWORD *)(result + 24LL * *(_QWORD *)(result - 8));
        if ( (_QWORD *)result != v18 )
        {
          do
          {
            v18 -= 3;
            sub_91D830(v18);
          }
          while ( *((_QWORD **)v11 + 2) != v18 );
        }
        v11 += 10;
        result = j_j_j___libc_free_0_0(v18 - 1);
        if ( v12 == v11 )
          goto LABEL_33;
      }
      else
      {
        result = sub_C338F0(v11 + 2);
LABEL_21:
        v11 += 10;
        if ( v12 == v11 )
          goto LABEL_33;
      }
    }
    v14 = *v11;
LABEL_16:
    if ( v14 != (_DWORD)v34
      || *((_BYTE *)v11 + 4) != BYTE4(v34)
      || (v19 = (_QWORD *)*((_QWORD *)v11 + 1), v19 != v35)
      || ((v20 = v11 + 2, (_QWORD *)v10 == v19) ? (v21 = sub_C3E590(v20)) : (v21 = sub_C33D00(v20)), !v21) )
    {
      v13 = *((_QWORD *)v11 + 4);
      if ( v13 )
      {
        sub_91D830((_QWORD *)(v13 + 24));
        sub_BD7260(v13);
        sub_BD2DD0(v13);
      }
    }
    goto LABEL_19;
  }
LABEL_33:
  if ( (_QWORD *)v10 == v35 )
  {
    if ( v36 )
    {
      v24 = 3LL * *(v36 - 1);
      for ( k = &v36[v24]; v36 != k; sub_91D830(k) )
        k -= 3;
      result = j_j_j___libc_free_0_0(k - 1);
    }
  }
  else
  {
    result = sub_C338F0(&v35);
  }
  if ( v10 != v32 )
    return sub_C338F0(&v32);
  if ( v33 )
  {
    v22 = 3LL * *(v33 - 1);
    for ( m = &v33[v22]; v33 != m; sub_91D830(m) )
      m -= 3;
    return j_j_j___libc_free_0_0(m - 1);
  }
  return result;
}
