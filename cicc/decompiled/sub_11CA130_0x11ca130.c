// Function: sub_11CA130
// Address: 0x11ca130
//
__int64 __fastcall sub_11CA130(__int64 a1, char a2, __int64 a3, __int64 *a4)
{
  __int64 *v6; // r13
  __int64 v7; // rax
  _QWORD v9[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v10[8]; // [rsp+20h] [rbp-40h] BYREF

  v6 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v7 = sub_BCD140(*(_QWORD **)(a3 + 72), *(_DWORD *)(*a4 + 172));
  v10[0] = a1;
  v9[0] = v6;
  v9[1] = v7;
  v10[1] = sub_AD64C0(v7, a2, 0);
  return sub_11C9AF0(0x1CCu, v6, v9, 2, (int)v10, 2, a3, a4, 0);
}
