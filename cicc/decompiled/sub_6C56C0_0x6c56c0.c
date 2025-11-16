// Function: sub_6C56C0
// Address: 0x6c56c0
//
__int64 __fastcall sub_6C56C0(__int64 a1, __int64 a2)
{
  __int64 v4; // [rsp+8h] [rbp-B8h] BYREF
  _BYTE v5[176]; // [rsp+10h] [rbp-B0h] BYREF

  sub_6E2250(v5, &v4, 4, 1, *(_QWORD *)(a1 + 16), a1);
  sub_6C55E0(0, a2 != 0, a2, 0, 0, (__int64 *)(a1 + 8));
  sub_6E2920(*(_QWORD *)(a1 + 8));
  return sub_6E2C70(v4, 1, *(_QWORD *)(a1 + 16), a1);
}
