// Function: sub_227A820
// Address: 0x227a820
//
_QWORD *__fastcall sub_227A820(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax

  v2 = (_QWORD *)sub_22077B0(0x10u);
  if ( v2 )
  {
    *v2 = &unk_4A08BA8;
    v2[1] = *(_QWORD *)(a2 + 8);
  }
  *a1 = v2;
  return a1;
}
