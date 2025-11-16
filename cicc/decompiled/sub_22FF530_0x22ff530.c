// Function: sub_22FF530
// Address: 0x22ff530
//
void __fastcall sub_22FF530(unsigned __int64 a1)
{
  __int64 *v2; // rdi

  v2 = (__int64 *)(a1 + 8);
  *(v2 - 1) = (__int64)&unk_4A0B150;
  sub_FDC110(v2);
  j_j___libc_free_0(a1);
}
