// Function: sub_2257710
// Address: 0x2257710
//
volatile signed __int32 *__fastcall sub_2257710(__int64 a1, volatile signed __int32 **a2)
{
  volatile signed __int32 **v2; // rdi

  v2 = (volatile signed __int32 **)(a1 + 8);
  *(v2 - 1) = (volatile signed __int32 *)off_4A082C8;
  return sub_2215E70(v2, a2);
}
