// Function: sub_15D1010
// Address: 0x15d1010
//
bool __fastcall sub_15D1010(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v5; // rax
  __int64 v6; // rdi
  __int64 *v7; // rax
  __int64 v8; // rdi
  int v9; // r13d
  __int64 v10; // rax
  __int64 v12[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13[6]; // [rsp+10h] [rbp-30h] BYREF

  v5 = a2[1];
  v6 = *a1;
  v12[0] = *a2;
  v12[1] = v5 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = sub_15D0EA0(v6, v12);
  v8 = *a1;
  v9 = *((_DWORD *)v7 + 4);
  v10 = a3[1];
  v13[0] = *a3;
  v13[1] = v10 & 0xFFFFFFFFFFFFFFF8LL;
  return v9 > *((_DWORD *)sub_15D0EA0(v8, v13) + 4);
}
