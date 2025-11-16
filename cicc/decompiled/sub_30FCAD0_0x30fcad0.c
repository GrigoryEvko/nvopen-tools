// Function: sub_30FCAD0
// Address: 0x30fcad0
//
_QWORD *__fastcall sub_30FCAD0(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v3; // r15
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  _QWORD *v6; // r12
  unsigned __int64 v7; // r14
  unsigned __int64 v8; // rdx
  _QWORD *v9; // rax
  char v10; // di
  __int64 v12; // rax

  v3 = a1 + 1;
  v4 = sub_22077B0(0x188u);
  v5 = *a2;
  v6 = (_QWORD *)a1[2];
  v7 = v4;
  *(_QWORD *)(v4 + 32) = *a2;
  qmemcpy((void *)(v4 + 40), a2 + 1, 0x160u);
  if ( !v6 )
  {
    v6 = a1 + 1;
    if ( v3 == (_QWORD *)a1[3] )
    {
      v10 = 1;
LABEL_11:
      sub_220F040(v10, v7, v6, a1 + 1);
      ++a1[5];
      return (_QWORD *)v7;
    }
LABEL_13:
    v12 = sub_220EF80((__int64)v6);
    if ( v5 <= *(_QWORD *)(v12 + 32) )
    {
      v6 = (_QWORD *)v12;
      goto LABEL_15;
    }
LABEL_9:
    v10 = 1;
    if ( v3 != v6 )
      v10 = v5 < v6[4];
    goto LABEL_11;
  }
  while ( 1 )
  {
    v8 = v6[4];
    v9 = (_QWORD *)v6[3];
    if ( v5 < v8 )
      v9 = (_QWORD *)v6[2];
    if ( !v9 )
      break;
    v6 = v9;
  }
  if ( v5 < v8 )
  {
    if ( v6 == (_QWORD *)a1[3] )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v8 < v5 )
    goto LABEL_9;
LABEL_15:
  j_j___libc_free_0(v7);
  return v6;
}
