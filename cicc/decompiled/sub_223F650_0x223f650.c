// Function: sub_223F650
// Address: 0x223f650
//
void __fastcall sub_223F650(_QWORD *a1)
{
  unsigned __int64 v1; // rbp
  unsigned __int64 v2; // rdi

  v1 = (unsigned __int64)a1 + *(_QWORD *)(*a1 - 24LL);
  v2 = *(_QWORD *)(v1 + 88);
  *(_QWORD *)(v1 + 120) = off_4A071A0;
  *(_QWORD *)v1 = off_4A07178;
  *(_QWORD *)(v1 + 16) = off_4A07080;
  if ( v2 != v1 + 104 )
    j___libc_free_0(v2);
  *(_QWORD *)(v1 + 16) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(v1 + 72));
  *(_QWORD *)(v1 + 8) = 0;
  *(_QWORD *)v1 = qword_4A07108;
  *(_QWORD *)(v1 + 120) = off_4A06798;
  sub_222E050(v1 + 120);
  j___libc_free_0(v1);
}
