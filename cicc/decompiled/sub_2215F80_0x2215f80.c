// Function: sub_2215F80
// Address: 0x2215f80
//
void *__fastcall sub_2215F80(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  void *result; // rax

  result = sub_2215EE0(a2, &a2[a3]);
  *a1 = result;
  return result;
}
