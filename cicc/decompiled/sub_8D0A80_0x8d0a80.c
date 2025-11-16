// Function: sub_8D0A80
// Address: 0x8d0a80
//
_QWORD **__fastcall sub_8D0A80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rbx
  _QWORD **result; // rax

  v6 = (_QWORD *)qword_4F60550;
  if ( qword_4F60550 )
    qword_4F60550 = *(_QWORD *)qword_4F60550;
  else
    v6 = (_QWORD *)sub_822B10(16, a2, a3, a4, a5, a6);
  *v6 = 0;
  v6[1] = a1;
  *v6 = unk_4D03FF8;
  if ( (_QWORD *)qword_4D03FF0 != a1 )
    sub_8D0910(a1);
  result = &qword_4D03FD0;
  if ( qword_4D03FD0 != a1 )
  {
    result = (_QWORD **)dword_4D03FC8;
    ++dword_4D03FC8[0];
  }
  unk_4D03FF8 = v6;
  return result;
}
