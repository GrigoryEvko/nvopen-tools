// Function: sub_1643EE0
// Address: 0x1643ee0
//
__int64 __fastcall sub_1643EE0(_QWORD *a1, __int64 *a2, unsigned int a3)
{
  __int64 v3; // rax

  v3 = *a2;
  a1[3] = a2;
  *a1 = v3;
  a1[4] = a3;
  a1[2] = a1 + 3;
  a1[1] = 0x100000010LL;
  return 0x100000010LL;
}
