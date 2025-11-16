// Function: sub_2E39C30
// Address: 0x2e39c30
//
void __fastcall sub_2E39C30(_QWORD *a1)
{
  __int64 *v2; // rdi

  v2 = a1 + 25;
  *(v2 - 25) = (__int64)&unk_4A288F0;
  sub_2E39BC0(v2);
  *a1 = &unk_49DAF80;
  sub_BB9100((__int64)a1);
  j_j___libc_free_0((unsigned __int64)a1);
}
