// Function: sub_11CB120
// Address: 0x11cb120
//
__int64 __fastcall sub_11CB120(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v9; // r14
  __int64 *v10; // rsi
  __int64 v11; // rax
  _QWORD v13[4]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v14[10]; // [rsp+30h] [rbp-50h] BYREF

  v9 = sub_BCE3C0(*(__int64 **)(a4 + 72), 0);
  v10 = (__int64 *)sub_BCD140(*(_QWORD **)(a4 + 72), *(_DWORD *)(*a5 + 172));
  v11 = *(_QWORD *)(a3 + 8);
  v14[0] = a1;
  v14[1] = a2;
  v14[2] = a3;
  v13[0] = v9;
  v13[1] = v9;
  v13[2] = v11;
  return sub_11C9AF0(0x207u, v10, v13, 3, (int)v14, 3, a4, a5, 0);
}
