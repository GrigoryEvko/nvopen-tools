// Function: sub_2F40530
// Address: 0x2f40530
//
_QWORD *__fastcall sub_2F40530(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 v6; // rax
  _QWORD *v7; // rbx

  v6 = sub_22077B0(0x60u);
  v7 = (_QWORD *)v6;
  if ( v6 )
  {
    sub_2F40450(v6, a3, a4);
    *v7 = &unk_4A2AED0;
  }
  *a1 = v7;
  return a1;
}
