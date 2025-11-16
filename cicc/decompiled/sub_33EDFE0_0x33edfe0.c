// Function: sub_33EDFE0
// Address: 0x33edfe0
//
_QWORD *__fastcall sub_33EDFE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // ebx
  __int64 *v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  int v10; // ebx
  __int64 v11; // r14
  __int64 v12; // rax
  unsigned int v13; // eax
  unsigned int v14; // r14d
  __int64 v15; // rsi
  __int64 v16; // rdx
  char v17; // al
  _QWORD v19[3]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+20h] [rbp-30h]
  __int64 v22; // [rsp+28h] [rbp-28h]

  v6 = a4;
  v19[0] = a2;
  v7 = *(__int64 **)(a1 + 64);
  v19[1] = a3;
  v8 = sub_3007410((__int64)v19, v7, a3, a4, a5, a6);
  v9 = v6;
  v10 = -1;
  v11 = v8;
  if ( v9 )
  {
    _BitScanReverse64(&v9, v9);
    v10 = 63 - (v9 ^ 0x3F);
  }
  v12 = sub_2E79000(*(__int64 **)(a1 + 40));
  v13 = sub_AE5260(v12, v11);
  if ( (unsigned __int8)v13 < (unsigned __int8)v10 )
    v13 = v10;
  v14 = v13;
  if ( LOWORD(v19[0]) )
  {
    if ( LOWORD(v19[0]) == 1 || (unsigned __int16)(LOWORD(v19[0]) - 504) <= 7u )
      BUG();
    v15 = *(_QWORD *)&byte_444C4A0[16 * LOWORD(v19[0]) - 16];
    v17 = byte_444C4A0[16 * LOWORD(v19[0]) - 8];
  }
  else
  {
    v21 = sub_3007260((__int64)v19);
    v15 = v21;
    v22 = v16;
    v17 = v16;
  }
  LOBYTE(v20) = v17;
  return sub_33EDE90(a1, (unsigned __int64)(v15 + 7) >> 3, v20, v14);
}
