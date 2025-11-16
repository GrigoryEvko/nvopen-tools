// Function: sub_223F5C0
// Address: 0x223f5c0
//
void __fastcall sub_223F5C0(unsigned __int64 a1)
{
  unsigned __int64 v2; // rdi

  *(_QWORD *)(a1 + 120) = off_4A071A0;
  *(_QWORD *)a1 = off_4A07178;
  *(_QWORD *)(a1 + 16) = off_4A07080;
  v2 = *(_QWORD *)(a1 + 88);
  if ( v2 != a1 + 104 )
    j___libc_free_0(v2);
  *(_QWORD *)(a1 + 16) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(a1 + 72));
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)a1 = qword_4A07108;
  *(_QWORD *)(a1 + 120) = off_4A06798;
  sub_222E050(a1 + 120);
  j___libc_free_0(a1);
}
