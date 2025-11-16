// Function: sub_223F820
// Address: 0x223f820
//
void __fastcall sub_223F820(_QWORD *a1)
{
  char *v1; // rbx
  unsigned __int64 v2; // rdi

  v1 = (char *)a1 + *(_QWORD *)(*a1 - 24LL);
  v2 = *((_QWORD *)v1 + 12);
  *(_QWORD *)v1 = off_4A073F0;
  *((_QWORD *)v1 + 2) = off_4A07418;
  *((_QWORD *)v1 + 16) = off_4A07440;
  *((_QWORD *)v1 + 3) = off_4A07080;
  if ( (char *)v2 != v1 + 112 )
    j___libc_free_0(v2);
  *((_QWORD *)v1 + 3) = off_4A07480;
  sub_2209150((volatile signed __int32 **)v1 + 10);
  *((_QWORD *)v1 + 1) = 0;
  *((_QWORD *)v1 + 2) = qword_4A07288;
  *(_QWORD *)v1 = qword_4A072D8;
  *((_QWORD *)v1 + 16) = off_4A06798;
  sub_222E050((__int64)(v1 + 128));
}
