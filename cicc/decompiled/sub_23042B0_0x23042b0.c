// Function: sub_23042B0
// Address: 0x23042b0
//
_QWORD *__fastcall sub_23042B0(_QWORD *a1)
{
  _QWORD *v1; // rax
  _BYTE v3[9]; // [rsp+Fh] [rbp-11h] BYREF

  sub_DF5000((__int64)v3);
  v1 = (_QWORD *)sub_22077B0(0x10u);
  if ( v1 )
    *v1 = &unk_4A15A78;
  *a1 = v1;
  return a1;
}
