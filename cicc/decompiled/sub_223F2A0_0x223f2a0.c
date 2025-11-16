// Function: sub_223F2A0
// Address: 0x223f2a0
//
void __fastcall sub_223F2A0(__int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)(a1 + 112) = off_4A07260;
  *(_QWORD *)a1 = off_4A07238;
  *(_QWORD *)(a1 + 8) = off_4A07080;
  v2 = *(_QWORD *)(a1 + 80);
  if ( v2 != a1 + 96 )
    j___libc_free_0(v2);
  *(_QWORD *)(a1 + 8) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 64));
  *(_QWORD *)a1 = qword_4A071C8;
  *(_QWORD *)(a1 + 112) = off_4A06798;
  sub_222E050(a1 + 112);
}
