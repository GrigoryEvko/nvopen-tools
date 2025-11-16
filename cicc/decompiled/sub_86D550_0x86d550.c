// Function: sub_86D550
// Address: 0x86d550
//
_BYTE *__fastcall sub_86D550(__int64 a1)
{
  _BYTE *v1; // r12
  __int64 v2; // rax

  sub_860260(a1);
  v1 = sub_732EF0(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v2 = sub_86B2C0(0);
  *(_QWORD *)(v2 + 24) = *(_QWORD *)&dword_4F063F8;
  *(_QWORD *)(v2 + 56) = qword_4F06BC0;
  sub_86CBE0(v2);
  return v1;
}
