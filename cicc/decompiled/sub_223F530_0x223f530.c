// Function: sub_223F530
// Address: 0x223f530
//
void __fastcall sub_223F530(_QWORD *a1)
{
  char *v1; // rbx
  unsigned __int64 v2; // rdi

  v1 = (char *)a1 + *(_QWORD *)(*a1 - 24LL);
  v2 = *((_QWORD *)v1 + 11);
  *((_QWORD *)v1 + 15) = off_4A071A0;
  *(_QWORD *)v1 = off_4A07178;
  *((_QWORD *)v1 + 2) = off_4A07080;
  if ( (char *)v2 != v1 + 104 )
    j___libc_free_0(v2);
  *((_QWORD *)v1 + 2) = off_4A07480;
  sub_2209150((volatile signed __int32 **)v1 + 9);
  *((_QWORD *)v1 + 1) = 0;
  *(_QWORD *)v1 = qword_4A07108;
  *((_QWORD *)v1 + 15) = off_4A06798;
  sub_222E050((__int64)(v1 + 120));
}
