// Function: sub_2C19E60
// Address: 0x2c19e60
//
unsigned __int64 *__fastcall sub_2C19E60(__int64 *a1)
{
  unsigned __int64 *v1; // r12
  unsigned __int64 v2; // rdx
  __int64 v3; // rax

  v1 = (unsigned __int64 *)a1[4];
  v2 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  *v1 = v2 | *v1 & 7;
  *(_QWORD *)(v2 + 8) = v1;
  v3 = *a1;
  a1[3] &= 7uLL;
  a1[4] = 0;
  (*(void (**)(void))(v3 + 8))();
  return v1;
}
