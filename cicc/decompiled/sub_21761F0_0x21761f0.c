// Function: sub_21761F0
// Address: 0x21761f0
//
__int64 __fastcall sub_21761F0(__int64 a1, unsigned int a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v14; // [rsp+0h] [rbp-30h] BYREF
  int v15; // [rsp+8h] [rbp-28h]

  v8 = *(_QWORD *)(a1 + 72);
  v14 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v14, v8, 2);
  v9 = (unsigned __int8 *)(*(_QWORD *)(a1 + 40) + 16LL * a2);
  v15 = *(_DWORD *)(a1 + 64);
  v10 = sub_1D2CCE0(
          a3,
          (*v9 != 5) + 4449,
          (__int64)&v14,
          *v9,
          *((_QWORD *)v9 + 1),
          a6,
          *(_OWORD *)*(_QWORD *)(a1 + 32),
          *(_OWORD *)(*(_QWORD *)(a1 + 32) + 40LL));
  v11 = v14;
  v12 = v10;
  *(_DWORD *)(v10 + 64) = *(_DWORD *)(a1 + 64);
  if ( v11 )
    sub_161E7C0((__int64)&v14, v11);
  return v12;
}
