// Function: sub_BBACF0
// Address: 0xbbacf0
//
_QWORD *__fastcall sub_BBACF0(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax

  v2 = (_QWORD *)sub_22077B0(16);
  if ( v2 )
  {
    *v2 = &unk_49DB0A8;
    v2[1] = *(_QWORD *)(a2 + 8);
  }
  *a1 = v2;
  return a1;
}
