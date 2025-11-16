// Function: sub_D21760
// Address: 0xd21760
//
__int64 __fastcall sub_D21760(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(__int64 *, __int64); // [rsp+18h] [rbp-28h]

  v7 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v8 = sub_BC0510(a4, &unk_4F86A90, a3);
  v10[0] = v7;
  v11 = sub_D1A5A0;
  v12 = sub_D1A4B0;
  sub_D216B0(a1, a3, (__int64)v10, (_QWORD *)(v8 + 8));
  if ( v11 )
    v11(v10, v10, 3);
  return a1;
}
