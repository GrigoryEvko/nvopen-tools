// Function: sub_22FF940
// Address: 0x22ff940
//
void __fastcall sub_22FF940(unsigned __int64 a1, __int64 a2)
{
  __int64 v3; // rdi

  v3 = a1 + 8;
  *(_QWORD *)(v3 - 8) = &unk_4A0B268;
  sub_D89DE0(v3, a2);
  j_j___libc_free_0(a1);
}
