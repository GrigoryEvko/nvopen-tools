// Function: sub_1444990
// Address: 0x1444990
//
_QWORD *__fastcall sub_1444990(_QWORD *a1, unsigned __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  unsigned __int64 v4; // rbx
  unsigned __int64 v5; // rdx
  _QWORD *v6; // rax
  _BOOL4 v7; // r8d
  __int64 v8; // rbx
  __int64 v10; // rax
  _BOOL4 v11; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v3 = (_QWORD *)a1[2];
  if ( !v3 )
  {
    v3 = a1 + 1;
    if ( v2 == (_QWORD *)a1[3] )
    {
      v7 = 1;
LABEL_11:
      v11 = v7;
      v8 = sub_22077B0(40);
      *(_QWORD *)(v8 + 32) = *a2;
      sub_220F040(v11, v8, v3, v2);
      ++a1[5];
      return (_QWORD *)v8;
    }
    v4 = *a2;
LABEL_13:
    v10 = sub_220EF80(v3);
    if ( v4 <= *(_QWORD *)(v10 + 32) )
      return (_QWORD *)v10;
LABEL_9:
    v7 = 1;
    if ( v2 != v3 )
      v7 = v4 < v3[4];
    goto LABEL_11;
  }
  v4 = *a2;
  while ( 1 )
  {
    v5 = v3[4];
    v6 = (_QWORD *)v3[3];
    if ( v4 < v5 )
      v6 = (_QWORD *)v3[2];
    if ( !v6 )
      break;
    v3 = v6;
  }
  if ( v4 < v5 )
  {
    if ( (_QWORD *)a1[3] == v3 )
      goto LABEL_9;
    goto LABEL_13;
  }
  if ( v4 > v5 )
    goto LABEL_9;
  return v3;
}
