// Function: sub_222AA70
// Address: 0x222aa70
//
void __fastcall sub_222AA70(unsigned __int64 a1)
{
  volatile signed __int32 **v2; // rdi

  v2 = (volatile signed __int32 **)(a1 + 56);
  *(v2 - 7) = (volatile signed __int32 *)off_4A07480;
  sub_2209150(v2);
  j___libc_free_0(a1);
}
