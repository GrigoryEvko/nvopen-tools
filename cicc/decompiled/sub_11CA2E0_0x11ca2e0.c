// Function: sub_11CA2E0
// Address: 0x11ca2e0
//
__int64 __fastcall sub_11CA2E0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v6; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v9[6]; // [rsp+10h] [rbp-30h] BYREF

  v6 = (__int64 *)sub_BCE3C0(*(__int64 **)(a3 + 72), 0);
  v9[0] = a1;
  v9[1] = a2;
  v8[0] = v6;
  v8[1] = v6;
  return sub_11C9AF0(0x1C8u, v6, v8, 2, (int)v9, 2, a3, a4, 0);
}
