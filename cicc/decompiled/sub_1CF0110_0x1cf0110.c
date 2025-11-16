// Function: sub_1CF0110
// Address: 0x1cf0110
//
__int64 __fastcall sub_1CF0110(int a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v9; // r13
  __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // r12

  v9 = sub_15E26F0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a7 + 40) + 56LL) + 40LL), a1, a2, a3);
  v10 = *(_QWORD *)(*(_QWORD *)v9 + 24LL);
  v11 = sub_1648AB0(72, (int)a5 + 1, 0);
  v12 = (__int64)v11;
  if ( v11 )
  {
    sub_15F1EA0((__int64)v11, **(_QWORD **)(v10 + 16), 54, (__int64)&v11[-3 * a5 - 3], a5 + 1, a7);
    *(_QWORD *)(v12 + 56) = 0;
    sub_15F5B40(v12, v10, v9, a4, a5, a6, 0, 0);
  }
  return v12;
}
