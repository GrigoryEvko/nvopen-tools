// Function: sub_BCACC0
// Address: 0xbcacc0
//
__int64 __fastcall sub_BCACC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rbx
  _QWORD *i; // rbx
  __int64 v13; // [rsp+0h] [rbp-40h] BYREF
  _QWORD *v14; // [rsp+8h] [rbp-38h]

  v5 = sub_BCAC60(a1, a2, a3, a4, a5);
  v9 = sub_C33340(a1, a2, v6, v7, v8);
  v10 = v9;
  if ( v5 != v9 )
  {
    sub_C373C0(&v13, v5, 0);
    if ( v13 != v10 )
      goto LABEL_3;
LABEL_7:
    sub_C3CEB0(&v13, 0);
    LOBYTE(v5) = v13 != v10;
    if ( v13 != v10 )
      goto LABEL_4;
    goto LABEL_8;
  }
  sub_C3C500(&v13, v9, 0);
  if ( v13 == v10 )
    goto LABEL_7;
LABEL_3:
  sub_C37310(&v13, 0);
  LOBYTE(v5) = v13 != v10;
  if ( v13 != v10 )
  {
LABEL_4:
    sub_C338F0(&v13);
    return (unsigned int)v5;
  }
LABEL_8:
  if ( !v14 )
    return (unsigned int)v5;
  for ( i = &v14[3 * *(v14 - 1)]; v14 != i; sub_91D830(i) )
    i -= 3;
  j_j_j___libc_free_0_0(i - 1);
  return (unsigned int)v5;
}
