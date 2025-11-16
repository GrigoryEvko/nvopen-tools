// Function: sub_22CFE50
// Address: 0x22cfe50
//
_QWORD *__fastcall sub_22CFE50(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rax

  v2 = (_QWORD *)sub_22077B0(0x10u);
  if ( v2 )
  {
    *v2 = &unk_4A09EA8;
    v2[1] = *(_QWORD *)(a2 + 8);
  }
  *a1 = v2;
  return a1;
}
