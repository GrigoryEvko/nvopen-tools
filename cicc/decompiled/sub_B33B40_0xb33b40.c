// Function: sub_B33B40
// Address: 0xb33b40
//
__int64 __fastcall sub_B33B40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int64 v8; // rsi
  __int64 v10; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v11[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v12; // [rsp+30h] [rbp-30h]

  v6 = *(_QWORD *)(a1 + 48);
  v10 = a2;
  v7 = sub_B6E160(*(_QWORD *)(*(_QWORD *)(v6 + 72) + 40LL), 11, 0, 0);
  v8 = 0;
  v12 = 257;
  if ( v7 )
    v8 = *(_QWORD *)(v7 + 24);
  return sub_B33530((unsigned int **)a1, v8, v7, (int)&v10, 1, (__int64)v11, a3, a4, 0);
}
