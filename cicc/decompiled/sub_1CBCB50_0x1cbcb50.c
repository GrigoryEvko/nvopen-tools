// Function: sub_1CBCB50
// Address: 0x1cbcb50
//
_QWORD *__fastcall sub_1CBCB50(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __int64 v6; // r13
  _QWORD *v7; // r12
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rax
  _BOOL8 v10; // rdi
  __int64 v12; // rax

  v3 = a1 + 1;
  v4 = sub_22077B0(48);
  v5 = *a2;
  v6 = v4;
  *(_QWORD *)(v4 + 32) = *a2;
  v7 = (_QWORD *)a1[2];
  *(_QWORD *)(v4 + 40) = a2[1];
  if ( !v7 )
  {
    v7 = a1 + 1;
    if ( v3 == (_QWORD *)a1[3] )
    {
      v10 = 1;
LABEL_11:
      sub_220F040(v10, v6, v7, a1 + 1);
      ++a1[5];
      return (_QWORD *)v6;
    }
LABEL_13:
    v12 = sub_220EF80(v7);
    if ( v5 <= *(_QWORD *)(v12 + 32) )
    {
      v7 = (_QWORD *)v12;
      goto LABEL_15;
    }
LABEL_9:
    v10 = 1;
    if ( v3 != v7 )
      v10 = v5 < v7[4];
    goto LABEL_11;
  }
  while ( 1 )
  {
    v8 = v7[4];
    v9 = (_QWORD *)v7[3];
    if ( v5 < v8 )
      v9 = (_QWORD *)v7[2];
    if ( !v9 )
      break;
    v7 = v9;
  }
  if ( v5 < v8 )
  {
    if ( v7 == (_QWORD *)a1[3] )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v8 < v5 )
    goto LABEL_9;
LABEL_15:
  j_j___libc_free_0(v6, 48);
  return v7;
}
