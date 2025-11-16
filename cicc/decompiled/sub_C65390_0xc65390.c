// Function: sub_C65390
// Address: 0xc65390
//
bool __fastcall sub_C65390(__int64 a1, const void *a2, __int64 a3)
{
  __int64 v4; // rdx

  v4 = *(_QWORD *)(a1 + 8);
  return v4 == a3 && memcmp(*(const void **)a1, a2, 4 * v4) == 0;
}
