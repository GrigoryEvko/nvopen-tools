// Function: sub_857470
// Address: 0x857470
//
void *__fastcall sub_857470(int a1)
{
  const void *v1; // r12
  void *v2; // rax
  size_t n[2]; // [rsp+8h] [rbp-18h] BYREF

  v1 = (const void *)sub_857250(a1, (__int64 *)n);
  v2 = (void *)sub_7279A0(n[0]);
  return memcpy(v2, v1, n[0]);
}
