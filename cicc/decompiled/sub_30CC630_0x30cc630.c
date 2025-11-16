// Function: sub_30CC630
// Address: 0x30cc630
//
_QWORD *__fastcall sub_30CC630(_QWORD *a1, __int64 a2, _QWORD *a3, char a4)
{
  __int64 v5; // r15
  __int64 v6; // rax
  _QWORD *v7; // rbx

  v5 = sub_30CC5F0(a2, (__int64)a3);
  v6 = sub_22077B0(0x40u);
  v7 = (_QWORD *)v6;
  if ( v6 )
  {
    sub_30CABE0(v6, a2, a3, v5, a4);
    *v7 = &off_49D8810;
  }
  *a1 = v7;
  return a1;
}
