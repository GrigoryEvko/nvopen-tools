// Function: sub_2304180
// Address: 0x2304180
//
_QWORD *__fastcall sub_2304180(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax

  v2 = *(_QWORD *)(a2 + 8);
  v3 = (_QWORD *)sub_22077B0(0x10u);
  if ( v3 )
  {
    v3[1] = v2;
    *v3 = &unk_4A0AC28;
  }
  *a1 = v3;
  return a1;
}
