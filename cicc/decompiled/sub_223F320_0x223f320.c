// Function: sub_223F320
// Address: 0x223f320
//
void __fastcall sub_223F320(_QWORD *a1)
{
  char *v1; // rbx
  unsigned __int64 v2; // rdi

  v1 = (char *)a1 + *(_QWORD *)(*a1 - 24LL);
  v2 = *((_QWORD *)v1 + 10);
  *((_QWORD *)v1 + 14) = off_4A07260;
  *(_QWORD *)v1 = off_4A07238;
  *((_QWORD *)v1 + 1) = off_4A07080;
  if ( (char *)v2 != v1 + 96 )
    j___libc_free_0(v2);
  *((_QWORD *)v1 + 1) = off_4A07480;
  sub_2209150((volatile signed __int32 **)v1 + 8);
  *(_QWORD *)v1 = qword_4A071C8;
  *((_QWORD *)v1 + 14) = off_4A06798;
  sub_222E050((__int64)(v1 + 112));
}
