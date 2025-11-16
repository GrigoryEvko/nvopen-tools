// Function: sub_C3C840
// Address: 0xc3c840
//
void *__fastcall sub_C3C840(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax

  *a1 = *a2;
  v2 = a2[1];
  a2[1] = 0;
  a1[1] = v2;
  *a2 = &unk_3F655C0;
  return &unk_3F655C0;
}
