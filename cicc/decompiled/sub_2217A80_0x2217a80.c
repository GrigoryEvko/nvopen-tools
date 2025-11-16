// Function: sub_2217A80
// Address: 0x2217a80
//
void __fastcall sub_2217A80(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  v2 = a1 + 16;
  *(_QWORD *)(v2 - 16) = off_4A05640;
  sub_2254270(v2);
  nullsub_801();
  j___libc_free_0(a1);
}
