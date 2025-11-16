// Function: sub_390CA60
// Address: 0x390ca60
//
__int64 __fastcall sub_390CA60(__int64 a1, __int64 **a2, __int64 a3)
{
  unsigned int v4; // r13d
  __int64 v5; // r14
  unsigned __int64 v7; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v8[4]; // [rsp+10h] [rbp-50h] BYREF
  int v9; // [rsp+30h] [rbp-30h]
  __int64 v10; // [rsp+38h] [rbp-28h]

  v4 = *(_DWORD *)(a3 + 64);
  v5 = **a2;
  sub_38CF260(*(_QWORD *)(a3 + 48), &v7, a2);
  *(_DWORD *)(a3 + 64) = 0;
  v9 = 1;
  v8[0] = &unk_49EFC48;
  v10 = a3 + 56;
  memset(&v8[1], 0, 24);
  sub_16E7A40((__int64)v8, 0, 0, 0);
  sub_38C6CD0(v5, v7, (__int64)v8);
  LOBYTE(v4) = *(_DWORD *)(a3 + 64) != v4;
  v8[0] = &unk_49EFD28;
  sub_16E7960((__int64)v8);
  return v4;
}
