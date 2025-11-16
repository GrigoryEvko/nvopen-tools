// Function: sub_384CF20
// Address: 0x384cf20
//
void __fastcall sub_384CF20(_QWORD *a1)
{
  _QWORD *v2; // rdi

  v2 = a1 + 20;
  *(v2 - 20) = off_4A3DBA8;
  *v2 = &unk_4A3DC60;
  sub_160F3F0((__int64)v2);
  sub_1636790(a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
