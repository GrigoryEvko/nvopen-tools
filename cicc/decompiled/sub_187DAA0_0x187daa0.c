// Function: sub_187DAA0
// Address: 0x187daa0
//
_QWORD *__fastcall sub_187DAA0(_QWORD *a1, unsigned __int64 *a2, _BYTE *a3)
{
  _QWORD *v5; // rax
  unsigned __int64 v6; // rbx
  bool v7; // zf
  _QWORD *v8; // r13
  _QWORD *v9; // r12
  _QWORD *v10; // r15
  unsigned __int64 v11; // rdx
  _QWORD *v12; // rax
  _BOOL8 v13; // rdi
  __int64 v15; // rax

  v5 = (_QWORD *)sub_22077B0(80);
  v6 = *a2;
  v7 = *a3 == 0;
  v8 = v5;
  v5[4] = *a2;
  if ( v7 )
  {
    v5[6] = 0;
    v5[5] = byte_3F871B3;
  }
  else
  {
    v5[5] = 0;
  }
  v9 = (_QWORD *)a1[2];
  v5[7] = 0;
  v10 = a1 + 1;
  v5[8] = 0;
  v5[9] = 0;
  if ( !v9 )
  {
    v9 = a1 + 1;
    if ( v10 == (_QWORD *)a1[3] )
    {
      v13 = 1;
      goto LABEL_12;
    }
LABEL_15:
    v15 = sub_220EF80(v9);
    if ( v6 <= *(_QWORD *)(v15 + 32) )
    {
      v9 = (_QWORD *)v15;
      goto LABEL_17;
    }
    if ( !v9 )
      goto LABEL_17;
    v13 = 1;
    if ( v10 != v9 )
      goto LABEL_20;
    goto LABEL_12;
  }
  while ( 1 )
  {
    v11 = v9[4];
    v12 = (_QWORD *)v9[3];
    if ( v11 > v6 )
      v12 = (_QWORD *)v9[2];
    if ( !v12 )
      break;
    v9 = v12;
  }
  if ( v6 < v11 )
  {
    if ( (_QWORD *)a1[3] != v9 )
      goto LABEL_15;
LABEL_11:
    v13 = 1;
    if ( v10 != v9 )
LABEL_20:
      v13 = v6 < v9[4];
LABEL_12:
    sub_220F040(v13, v8, v9, a1 + 1);
    ++a1[5];
    return v8;
  }
  if ( v11 < v6 )
    goto LABEL_11;
LABEL_17:
  j_j___libc_free_0(v8, 80);
  return v9;
}
