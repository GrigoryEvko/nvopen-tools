// Function: sub_1E63F40
// Address: 0x1e63f40
//
_QWORD *__fastcall sub_1E63F40(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v4; // r12
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // rdx
  _QWORD *v7; // rax
  _BOOL4 v8; // r8d
  __int64 v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  _BOOL4 v13; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v4 = (_QWORD *)a1[2];
  if ( !v4 )
  {
    v4 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v8 = 1;
LABEL_11:
      v13 = v8;
      v9 = sub_22077B0(48);
      *(_QWORD *)(v9 + 32) = *a2;
      v10 = a2[1];
      a2[1] = 0;
      *(_QWORD *)(v9 + 40) = v10;
      sub_220F040(v13, v9, v4, v2);
      ++a1[5];
      return (_QWORD *)v9;
    }
    v5 = *a2;
LABEL_13:
    v12 = sub_220EF80(v4);
    if ( v5 <= *(_QWORD *)(v12 + 32) )
      return (_QWORD *)v12;
LABEL_9:
    v8 = 1;
    if ( v2 != v4 )
      v8 = v5 < v4[4];
    goto LABEL_11;
  }
  v5 = *a2;
  while ( 1 )
  {
    v6 = v4[4];
    v7 = (_QWORD *)v4[3];
    if ( v5 < v6 )
      v7 = (_QWORD *)v4[2];
    if ( !v7 )
      break;
    v4 = v7;
  }
  if ( v5 < v6 )
  {
    if ( v4 == (_QWORD *)a1[3] )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v5 > v6 )
    goto LABEL_9;
  return v4;
}
