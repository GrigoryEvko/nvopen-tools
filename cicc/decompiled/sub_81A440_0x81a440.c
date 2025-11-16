// Function: sub_81A440
// Address: 0x81a440
//
_QWORD *__fastcall sub_81A440(void *src, size_t n)
{
  _QWORD *v2; // r12
  void *v3; // rax

  v2 = (_QWORD *)sub_823970(24);
  *v2 = qword_4F19408;
  qword_4F19408 = (__int64)v2;
  v3 = (void *)sub_823970(n + 1);
  v2[1] = v3;
  memcpy(v3, src, n);
  *(_BYTE *)(v2[1] + n) = 0;
  v2[2] = 0;
  return v2;
}
