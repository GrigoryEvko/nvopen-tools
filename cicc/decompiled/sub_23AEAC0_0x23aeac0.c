// Function: sub_23AEAC0
// Address: 0x23aeac0
//
_QWORD *__fastcall sub_23AEAC0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax

  v3 = sub_3148040(a3, 0);
  v4 = (_QWORD *)sub_22077B0(0x10u);
  if ( v4 )
  {
    v4[1] = v3;
    *v4 = &unk_4A161C8;
  }
  *a1 = v4;
  return a1;
}
