// Function: sub_10611B0
// Address: 0x10611b0
//
__int64 __fastcall sub_10611B0(unsigned int ***a1, int a2, __int64 a3, const void *a4, unsigned __int64 a5, __int64 a6)
{
  unsigned int v10; // eax

  v10 = sub_B5AE90(a2);
  if ( v10 )
    return sub_1060DF0(a1, v10, a3, a4, a5, a6);
  sub_1060D20((__int64)a1, "No VPIntrinsic for this opcode");
  return 0;
}
