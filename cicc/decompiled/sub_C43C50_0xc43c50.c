// Function: sub_C43C50
// Address: 0xc43c50
//
bool __fastcall sub_C43C50(__int64 a1, const void **a2)
{
  size_t v2; // rdx

  v2 = 8 * (((unsigned __int64)*(unsigned int *)(a1 + 8) + 63) >> 6);
  return !v2 || memcmp(*(const void **)a1, *a2, v2) == 0;
}
