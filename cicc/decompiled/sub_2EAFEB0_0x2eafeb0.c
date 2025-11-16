// Function: sub_2EAFEB0
// Address: 0x2eafeb0
//
_QWORD *__fastcall sub_2EAFEB0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax

  v2 = (_QWORD *)sub_22077B0(0x10u);
  if ( v2 )
  {
    *v2 = &unk_4A29888;
    v2[1] = *(_QWORD *)(a2 + 8);
  }
  *a1 = v2;
  return a1;
}
