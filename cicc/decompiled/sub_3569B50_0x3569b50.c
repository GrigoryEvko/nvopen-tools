// Function: sub_3569B50
// Address: 0x3569b50
//
_QWORD *__fastcall sub_3569B50(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *v2; // r15
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  _QWORD *result; // rax
  __int64 v9; // rax
  __int64 v10; // r14
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  char v13; // di
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi

  v2 = a1 + 9;
  v4 = (__int64)(a1 + 9);
  v5 = (_QWORD *)a1[10];
  if ( v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = v5[2];
        v7 = v5[3];
        if ( v5[4] >= a2 )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v4 = (__int64)v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v6 );
LABEL_6:
    if ( v2 != (_QWORD *)v4 && *(_QWORD *)(v4 + 32) <= a2 )
      return *(_QWORD **)(v4 + 40);
  }
  v9 = sub_22077B0(0x30u);
  *(_QWORD *)(v9 + 32) = a2;
  v10 = v9;
  *(_QWORD *)(v9 + 40) = 0;
  v11 = sub_3569A50(a1 + 8, v4, (unsigned __int64 *)(v9 + 32));
  if ( v12 )
  {
    v13 = v2 == v12 || v11 || a2 < v12[4];
    sub_220F040(v13, v10, v12, v2);
    ++a1[13];
  }
  else
  {
    v15 = v10;
    v10 = (__int64)v11;
    j_j___libc_free_0(v15);
  }
  result = (_QWORD *)sub_22077B0(0x10u);
  if ( result )
  {
    result[1] = a1;
    *result = a2 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v14 = *(_QWORD *)(v10 + 40);
  *(_QWORD *)(v10 + 40) = result;
  if ( v14 )
  {
    j_j___libc_free_0(v14);
    return *(_QWORD **)(v10 + 40);
  }
  return result;
}
