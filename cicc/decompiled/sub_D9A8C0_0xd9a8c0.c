// Function: sub_D9A8C0
// Address: 0xd9a8c0
//
__int64 __fastcall sub_D9A8C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v10; // [rsp+8h] [rbp-38h]

  v10 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v6 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  sub_D98CB0(a1, a3, v10 + 8, v6 + 8, v7 + 8, v8 + 8);
  return a1;
}
