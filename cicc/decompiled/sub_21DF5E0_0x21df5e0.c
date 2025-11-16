// Function: sub_21DF5E0
// Address: 0x21df5e0
//
__int64 __fastcall sub_21DF5E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _QWORD *v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  int v13; // [rsp+8h] [rbp-28h]

  if ( *(_DWORD *)(*(_QWORD *)(a1 + 32) + 252LL) <= 0x45u )
    sub_16BD130("match instruction not supported on this architecture", 1u);
  v7 = *(_QWORD *)(a2 + 72);
  v8 = *(_QWORD **)(a1 - 176);
  v12 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v12, v7, 2);
  v9 = *(_QWORD *)(a2 + 32);
  v13 = *(_DWORD *)(a2 + 64);
  v10 = sub_1D23DE0(
          v8,
          2 * (**(_BYTE **)(*(_QWORD *)(v9 + 40) + 40LL) != 5) + 3137,
          (__int64)&v12,
          *(_QWORD *)(a2 + 40),
          *(_DWORD *)(a2 + 60),
          a6,
          (__int64 *)(v9 + 40),
          1);
  if ( v12 )
    sub_161E7C0((__int64)&v12, v12);
  return v10;
}
