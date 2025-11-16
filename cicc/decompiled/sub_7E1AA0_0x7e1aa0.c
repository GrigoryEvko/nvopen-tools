// Function: sub_7E1AA0
// Address: 0x7e1aa0
//
_QWORD *sub_7E1AA0()
{
  _QWORD *v0; // rax
  _QWORD *v1; // rcx
  __int64 v2; // rdi
  _QWORD *result; // rax
  _QWORD *v4; // rdx
  __int64 v5; // rdx

  v0 = qword_4D03F68;
  v1 = (_QWORD *)*qword_4D03F68;
  if ( !*((_BYTE *)qword_4D03F68 + 24) )
  {
    if ( !v1 )
      goto LABEL_4;
    v1[5] = qword_4D03F68[5];
    v1[6] = v0[6];
    if ( v1[1] != v0[1] )
      goto LABEL_4;
LABEL_10:
    v5 = v0[9];
    qword_4D03F68 = v1;
    v1[9] = v5;
    result = (_QWORD *)v0[10];
    v1[10] = result;
    return result;
  }
  qword_4F06BC0 = qword_4D03F68[7];
  if ( v1 && v1[1] == qword_4D03F68[1] )
    goto LABEL_10;
LABEL_4:
  v2 = v0[9];
  result = (_QWORD *)v2;
  if ( v2 )
  {
    do
    {
      v4 = result;
      result = (_QWORD *)*result;
    }
    while ( result );
    result = (_QWORD *)qword_4F18A18;
    *v4 = qword_4F18A18;
    qword_4F18A18 = v2;
  }
  qword_4D03F68 = v1;
  return result;
}
