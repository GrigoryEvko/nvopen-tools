// Function: sub_223F1A0
// Address: 0x223f1a0
//
void __fastcall sub_223F1A0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)a1 = off_4A07080;
  v2 = *(_QWORD *)(a1 + 72);
  if ( v2 != a1 + 88 )
    j___libc_free_0(v2);
  *(_QWORD *)a1 = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 56));
}
