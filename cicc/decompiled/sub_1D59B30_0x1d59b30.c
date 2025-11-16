// Function: sub_1D59B30
// Address: 0x1d59b30
//
bool *__fastcall sub_1D59B30(bool *a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // edx

  v3 = *(_DWORD *)(a3 + 8);
  a1[1] = 1;
  *a1 = v3 >> 8 == 1;
  return a1;
}
