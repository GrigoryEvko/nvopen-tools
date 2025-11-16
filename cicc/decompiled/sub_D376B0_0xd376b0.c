// Function: sub_D376B0
// Address: 0xd376b0
//
__int64 __fastcall sub_D376B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rbx
  __int64 v10; // [rsp+10h] [rbp-40h]
  __int64 v11; // [rsp+18h] [rbp-38h]

  v6 = sub_BC1CD0(a4, &unk_4F881D0, a3);
  v7 = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v10 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v11 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v8 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  *(_QWORD *)(a1 + 72) = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = v6 + 8;
  *(_QWORD *)(a1 + 40) = v7;
  *(_QWORD *)(a1 + 48) = v10 + 8;
  *(_QWORD *)(a1 + 56) = v11 + 8;
  *(_QWORD *)(a1 + 64) = v8 + 8;
  return a1;
}
