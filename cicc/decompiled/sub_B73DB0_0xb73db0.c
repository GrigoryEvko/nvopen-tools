// Function: sub_B73DB0
// Address: 0xb73db0
//
__int64 __fastcall sub_B73DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rbx
  __int64 *v12; // r15
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rax
  char v16; // al
  char v17; // al
  _QWORD *v18; // r13
  __int64 v19; // rax
  _QWORD *i; // r12
  __int64 v21; // rax
  _QWORD *j; // rbx
  __int64 *v23; // [rsp-80h] [rbp-80h]
  __int64 v24; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v25; // [rsp-70h] [rbp-70h]
  __int64 v26; // [rsp-58h] [rbp-58h] BYREF
  _QWORD *v27; // [rsp-50h] [rbp-50h]

  result = *(unsigned int *)(a1 + 24);
  if ( !(_DWORD)result )
    return result;
  v6 = sub_C33690(a1, a2, a3, a4, a5);
  v10 = sub_C33340(a1, a2, v7, v8, v9);
  v11 = v10;
  if ( v6 == v10 )
  {
    sub_C3C5A0(&v24, v10, 1);
    sub_C3C5A0(&v26, v11, 2);
  }
  else
  {
    sub_C36740(&v24, v6, 1);
    sub_C36740(&v26, v6, 2);
  }
  v12 = *(__int64 **)(a1 + 8);
  v13 = 4LL * *(unsigned int *)(a1 + 24);
  result = (__int64)&v12[v13];
  v23 = &v12[v13];
  if ( v12 != &v12[v13] )
  {
    while ( 1 )
    {
      v15 = *v12;
      if ( *v12 == v24 )
      {
        if ( v11 == v15 )
          v16 = sub_C3E590(v12);
        else
          v16 = sub_C33D00(v12);
        if ( v16 )
          goto LABEL_9;
        v15 = *v12;
        if ( v26 == *v12 )
          goto LABEL_17;
LABEL_7:
        v14 = v12[3];
        if ( v14 )
        {
          sub_91D830((_QWORD *)(v14 + 24));
          sub_BD7260(v14);
          sub_BD2DD0(v14);
        }
LABEL_9:
        if ( v11 == *v12 )
          goto LABEL_21;
LABEL_10:
        result = sub_C338F0(v12);
LABEL_11:
        v12 += 4;
        if ( v23 == v12 )
          break;
      }
      else
      {
        if ( v26 != v15 )
          goto LABEL_7;
LABEL_17:
        if ( v11 == v15 )
          v17 = sub_C3E590(v12);
        else
          v17 = sub_C33D00(v12);
        if ( !v17 )
          goto LABEL_7;
        if ( v11 != *v12 )
          goto LABEL_10;
LABEL_21:
        result = v12[1];
        if ( !result )
          goto LABEL_11;
        v18 = (_QWORD *)(result + 24LL * *(_QWORD *)(result - 8));
        if ( (_QWORD *)result != v18 )
        {
          do
          {
            v18 -= 3;
            sub_91D830(v18);
          }
          while ( (_QWORD *)v12[1] != v18 );
        }
        v12 += 4;
        result = j_j_j___libc_free_0_0(v18 - 1);
        if ( v23 == v12 )
          break;
      }
    }
  }
  if ( v11 != v26 )
  {
    result = sub_C338F0(&v26);
    goto LABEL_27;
  }
  if ( !v27 )
  {
LABEL_27:
    if ( v11 != v24 )
      return sub_C338F0(&v24);
    goto LABEL_36;
  }
  v19 = 3LL * *(v27 - 1);
  for ( i = &v27[v19]; v27 != i; sub_91D830(i) )
    i -= 3;
  result = j_j_j___libc_free_0_0(i - 1);
  if ( v11 != v24 )
    return sub_C338F0(&v24);
LABEL_36:
  if ( v25 )
  {
    v21 = 3LL * *(v25 - 1);
    for ( j = &v25[v21]; v25 != j; sub_91D830(j) )
      j -= 3;
    return j_j_j___libc_free_0_0(j - 1);
  }
  return result;
}
