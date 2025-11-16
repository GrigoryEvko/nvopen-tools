// Function: sub_305D470
// Address: 0x305d470
//
_QWORD *__fastcall sub_305D470(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  _QWORD *v4; // rax

  v3 = sub_B2BEC0(a3);
  v4 = (_QWORD *)sub_22077B0(0x28u);
  if ( v4 )
  {
    v4[2] = v3;
    v4[3] = a2 + 1288;
    v4[4] = a2 + 2248;
    *v4 = &unk_4A31150;
    v4[1] = &unk_4A308A0;
  }
  *a1 = v4;
  return a1;
}
