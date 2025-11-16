// Function: sub_2304B20
// Address: 0x2304b20
//
_QWORD *__fastcall sub_2304B20(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rbx
  _QWORD *v6; // rax

  v5 = sub_227F6E0(a2 + 8, a3, a4, a5);
  v6 = (_QWORD *)sub_22077B0(0x10u);
  if ( v6 )
  {
    v6[1] = v5;
    *v6 = &unk_4A0B6C0;
  }
  *a1 = v6;
  return a1;
}
