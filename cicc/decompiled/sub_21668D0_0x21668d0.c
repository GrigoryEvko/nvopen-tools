// Function: sub_21668D0
// Address: 0x21668d0
//
void __fastcall sub_21668D0(__int64 a1)
{
  __int64 v1; // rdx
  __int64 v2; // rdx
  _QWORD *v3; // rax
  __int64 v4[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v5[6]; // [rsp+10h] [rbp-30h] BYREF

  sub_1F46F00(a1, &unk_4FC9074, 1, 1, 0);
  sub_1F46F00(a1, &unk_4FC4534, 1, 1, 0);
  sub_1F46F00(a1, &unk_4FC6A0C, 1, 1, 0);
  if ( dword_4FD26A0 == 1 )
  {
    sub_1F46F00(a1, &unk_4FCE24C, 1, 1, 0);
  }
  else
  {
    sub_1F46F00(a1, &unk_4FC8A0C, 1, 1, 1u);
    sub_1F46F00(a1, &unk_4FCE24C, 1, 1, 1u);
    sub_1F46F00(a1, &unk_4FC9D8C, 1, 1, 0);
  }
  if ( sub_1F46F00(a1, &unk_4FC7874, 1, 1, 0) )
  {
    v4[0] = (__int64)v5;
    sub_2165CE0(v4, "After Machine Scheduling", (__int64)"");
    sub_1F46460(a1, (__int64)v4, v1);
    if ( (_QWORD *)v4[0] != v5 )
      j_j___libc_free_0(v4[0], v5[0] + 1LL);
  }
  if ( !byte_4FD25C0 && (unsigned int)sub_1F45DD0(a1) )
  {
    v3 = (_QWORD *)sub_21F9D90();
    sub_1F46490(a1, v3, 1, 1, 0);
  }
  sub_1F46F00(a1, &unk_4FCAC8C, 1, 1, 0);
  v4[0] = (__int64)v5;
  sub_2165CE0(v4, "After StackSlotColoring", (__int64)"");
  sub_1F46460(a1, (__int64)v4, v2);
  if ( (_QWORD *)v4[0] != v5 )
    j_j___libc_free_0(v4[0], v5[0] + 1LL);
}
