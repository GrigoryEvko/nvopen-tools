// Function: sub_23AEA70
// Address: 0x23aea70
//
_QWORD *__fastcall sub_23AEA70(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax

  v3 = sub_3147DF0(a3, 0);
  v4 = (_QWORD *)sub_22077B0(0x10u);
  if ( v4 )
  {
    v4[1] = v3;
    *v4 = &unk_4A161F0;
  }
  *a1 = v4;
  return a1;
}
