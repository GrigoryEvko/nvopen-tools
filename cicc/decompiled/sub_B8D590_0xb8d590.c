// Function: sub_B8D590
// Address: 0xb8d590
//
__int64 **__fastcall sub_B8D590(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 *v3; // rcx
  __int64 v5; // rdx

  v2 = *((unsigned int *)a2 + 6);
  v3 = (__int64 *)*a2;
  *a1 = a2;
  v5 = a2[1] + 32 * v2;
  a1[1] = v3;
  a1[2] = (__int64 *)v5;
  a1[3] = (__int64 *)v5;
  return a1;
}
