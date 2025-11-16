// Function: sub_13453B0
// Address: 0x13453b0
//
int __fastcall sub_13453B0(_BYTE *a1, __int64 *a2, unsigned int *a3, __int64 a4, __int64 *a5)
{
  a5[1] &= 0xFFFFFFFFFFFFF000LL;
  *a5 &= ~0x8000uLL;
  return sub_13451C0(a1, a2, a3, a4, a5);
}
