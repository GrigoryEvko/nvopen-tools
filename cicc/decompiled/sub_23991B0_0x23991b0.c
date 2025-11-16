// Function: sub_23991B0
// Address: 0x23991b0
//
void *__fastcall sub_23991B0(_QWORD *a1)
{
  __int64 v1; // rbx
  void *result; // rax

  v1 = a1[1];
  result = &unk_4A0AAE8;
  *a1 = &unk_4A0AAE8;
  if ( v1 )
  {
    sub_2398D90(v1 + 64);
    return (void *)sub_2398F30(v1 + 32);
  }
  return result;
}
