// Function: sub_732B10
// Address: 0x732b10
//
_BYTE *__fastcall sub_732B10(__int64 a1)
{
  _BYTE *v1; // rax
  _BYTE *v2; // r12

  v1 = sub_726B30(0);
  *((_QWORD *)v1 + 6) = a1;
  v2 = v1;
  sub_7304E0(a1);
  return v2;
}
