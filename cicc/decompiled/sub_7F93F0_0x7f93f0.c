// Function: sub_7F93F0
// Address: 0x7f93f0
//
_QWORD **__fastcall sub_7F93F0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rcx
  _QWORD *v3; // rdx

  v1 = (_QWORD *)a1[5];
  if ( v1 )
  {
    v2 = qword_4D03F78;
    while ( 1 )
    {
      v3 = (_QWORD *)*v1;
      *v1 = v2;
      v2 = v1;
      qword_4D03F78 = v1;
      if ( !v3 )
        break;
      v1 = v3;
    }
  }
  *a1 = qword_4D03F70;
  qword_4D03F70 = a1;
  return &qword_4D03F70;
}
