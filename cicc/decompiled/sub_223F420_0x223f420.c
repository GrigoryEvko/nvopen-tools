// Function: sub_223F420
// Address: 0x223f420
//
void __fastcall sub_223F420(_QWORD *a1)
{
  unsigned __int64 v1; // rbp
  unsigned __int64 v2; // rdi

  v1 = (unsigned __int64)a1 + *(_QWORD *)(*a1 - 24LL);
  v2 = *(_QWORD *)(v1 + 80);
  *(_QWORD *)(v1 + 112) = off_4A07260;
  *(_QWORD *)v1 = off_4A07238;
  *(_QWORD *)(v1 + 8) = off_4A07080;
  if ( v2 != v1 + 96 )
    j___libc_free_0(v2);
  *(_QWORD *)(v1 + 8) = off_4A07480;
  sub_2209150((volatile signed __int32 **)(v1 + 64));
  *(_QWORD *)v1 = qword_4A071C8;
  *(_QWORD *)(v1 + 112) = off_4A06798;
  sub_222E050(v1 + 112);
  j___libc_free_0(v1);
}
