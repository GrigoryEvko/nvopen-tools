// Function: sub_2B3E050
// Address: 0x2b3e050
//
_QWORD *__fastcall sub_2B3E050(_QWORD *a1, unsigned int *a2, _DWORD *a3)
{
  _QWORD *v4; // r15
  __int64 v6; // rax
  __int64 v7; // rbx
  unsigned __int64 v8; // r13
  _QWORD *v9; // r12
  __int64 v10; // rdx
  _QWORD *v11; // rax
  char v12; // di
  __int64 v14; // rax

  v4 = a1 + 1;
  v6 = sub_22077B0(0x30u);
  v7 = *a2;
  v8 = v6;
  *(_QWORD *)(v6 + 32) = v7;
  LODWORD(v6) = *a3;
  v9 = (_QWORD *)a1[2];
  *(_DWORD *)(v8 + 40) = v6;
  if ( !v9 )
  {
    v9 = a1 + 1;
    if ( v4 == (_QWORD *)a1[3] )
    {
      v12 = 1;
      goto LABEL_10;
    }
LABEL_12:
    v14 = sub_220EF80((__int64)v9);
    if ( v7 <= *(_QWORD *)(v14 + 32) )
    {
      v9 = (_QWORD *)v14;
      goto LABEL_14;
    }
    if ( !v9 )
      goto LABEL_14;
    v12 = 1;
    if ( v9 != v4 )
      goto LABEL_17;
    goto LABEL_10;
  }
  while ( 1 )
  {
    v10 = v9[4];
    v11 = (_QWORD *)v9[3];
    if ( v7 < v10 )
      v11 = (_QWORD *)v9[2];
    if ( !v11 )
      break;
    v9 = v11;
  }
  if ( v7 < v10 )
  {
    if ( v9 != (_QWORD *)a1[3] )
      goto LABEL_12;
LABEL_9:
    v12 = 1;
    if ( v9 != v4 )
LABEL_17:
      v12 = v7 < v9[4];
LABEL_10:
    sub_220F040(v12, v8, v9, a1 + 1);
    ++a1[5];
    return (_QWORD *)v8;
  }
  if ( v7 > v10 )
    goto LABEL_9;
LABEL_14:
  j_j___libc_free_0(v8);
  return v9;
}
