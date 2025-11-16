// Function: sub_2166ED0
// Address: 0x2166ed0
//
void __fastcall sub_2166ED0(__int64 a1)
{
  _QWORD *v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v7[6]; // [rsp+10h] [rbp-30h] BYREF

  v1 = (_QWORD *)sub_22059C0();
  sub_1F46490(a1, v1, 1, 1, 0);
  if ( sub_1F46F00(a1, &unk_4FCAE50, 1, 1, 0) )
  {
    v6[0] = (__int64)v7;
    sub_2165CE0(v6, "After Pre-RegAlloc TailDuplicate", (__int64)"");
    sub_1F46460(a1, (__int64)v6, v2);
    if ( (_QWORD *)v6[0] != v7 )
      j_j___libc_free_0(v6[0], v7[0] + 1LL);
  }
  sub_1F46F00(a1, &unk_4FC83EC, 1, 1, 0);
  sub_1F46F00(a1, &unk_4FCA83C, 1, 1, 1u);
  sub_1F46F00(a1, &unk_4FC453C, 1, 1, 0);
  sub_1F46F00(a1, &unk_4FC332C, 1, 1, 0);
  v6[0] = (__int64)v7;
  sub_2165CE0(v6, "After codegen DCE pass", (__int64)"");
  sub_1F46460(a1, (__int64)v6, v3);
  if ( (_QWORD *)v6[0] != v7 )
    j_j___libc_free_0(v6[0], v7[0] + 1LL);
  if ( byte_4FD1980 )
    sub_1F46F00(a1, &unk_4FC64C8, 1, 1, 0);
  if ( byte_4FD18A0 )
    sub_1F46F00(a1, &unk_4FC5C94, 1, 1, 0);
  if ( byte_4FD1A60 )
    sub_1F46F00(a1, &unk_4FC7F74, 1, 1, 0);
  v6[0] = (__int64)v7;
  sub_2165CE0(v6, "After Machine LICM, CSE and Sinking passes", (__int64)"");
  sub_1F46460(a1, (__int64)v6, v4);
  if ( (_QWORD *)v6[0] != v7 )
    j_j___libc_free_0(v6[0], v7[0] + 1LL);
  sub_1F46F00(a1, &unk_4FC84D4, 1, 1, 0);
  v6[0] = (__int64)v7;
  sub_2165CE0(v6, "After codegen peephole optimization pass", (__int64)"");
  sub_1F46460(a1, (__int64)v6, v5);
  if ( (_QWORD *)v6[0] != v7 )
    j_j___libc_free_0(v6[0], v7[0] + 1LL);
}
