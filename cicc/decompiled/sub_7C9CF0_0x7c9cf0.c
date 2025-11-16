// Function: sub_7C9CF0
// Address: 0x7c9cf0
//
_BYTE *sub_7C9CF0()
{
  _QWORD *v0; // rax
  _BYTE *result; // rax

  v0 = sub_7247C0(qword_4F06C40 + 1LL);
  result = memcpy(v0, qword_4F06C50, qword_4F06C40);
  result[qword_4F06C40] = 0;
  return result;
}
