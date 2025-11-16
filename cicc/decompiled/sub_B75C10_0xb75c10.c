// Function: sub_B75C10
// Address: 0xb75c10
//
__int64 __fastcall sub_B75C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r13
  _QWORD *v13; // rdi
  __int64 result; // rax
  _QWORD *j; // rbx
  _QWORD *i; // rbx
  __int64 v17; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+20h] [rbp-50h]
  __int64 v20; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v21; // [rsp+30h] [rbp-40h]

  *(_QWORD *)(a1 + 16) = 0;
  v5 = sub_C33690(a1, a2, a3, a4, a5);
  v9 = sub_C33340(a1, a2, v6, v7, v8);
  v10 = v9;
  if ( v5 == v9 )
    sub_C3C5A0(&v17, v9, 1);
  else
    sub_C36740(&v17, v5, 1);
  LODWORD(v19) = -1;
  BYTE4(v19) = 1;
  if ( v17 == v10 )
    sub_C3C840(&v20, &v17);
  else
    sub_C338E0(&v20, &v17);
  if ( v17 == v10 )
  {
    if ( v18 )
    {
      for ( i = &v18[3 * *(v18 - 1)]; v18 != i; sub_91D830(i) )
        i -= 3;
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v17);
  }
  v11 = *(_QWORD **)(a1 + 8);
  v12 = &v11[5 * *(unsigned int *)(a1 + 24)];
  if ( v12 != v11 )
  {
    while ( 1 )
    {
      if ( !v11 )
        goto LABEL_10;
      v13 = v11 + 1;
      *v11 = v19;
      if ( v10 == v20 )
      {
        sub_C3C790(v13, &v20);
        v11 += 5;
        if ( v11 == v12 )
          break;
      }
      else
      {
        sub_C33EB0(v13, &v20);
LABEL_10:
        v11 += 5;
        if ( v11 == v12 )
          break;
      }
    }
  }
  if ( v10 != v20 )
    return sub_C338F0(&v20);
  result = (__int64)v21;
  if ( v21 )
  {
    for ( j = &v21[3 * *(v21 - 1)]; v21 != j; sub_91D830(j) )
      j -= 3;
    return j_j_j___libc_free_0_0(j - 1);
  }
  return result;
}
