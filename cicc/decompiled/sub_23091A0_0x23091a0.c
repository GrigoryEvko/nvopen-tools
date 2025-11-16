// Function: sub_23091A0
// Address: 0x23091a0
//
_QWORD *__fastcall sub_23091A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // rax

  v6 = (_QWORD *)sub_22077B0(0x20u);
  if ( v6 )
  {
    v6[1] = a3;
    v6[2] = a4;
    v6[3] = 0;
    *v6 = &unk_4A0B3D0;
  }
  *a1 = v6;
  return a1;
}
