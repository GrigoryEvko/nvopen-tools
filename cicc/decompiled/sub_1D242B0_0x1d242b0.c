// Function: sub_1D242B0
// Address: 0x1d242b0
//
_QWORD *__fastcall sub_1D242B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5, int a6)
{
  __int64 v10; // rsi
  _QWORD *v11; // rax
  unsigned __int8 *v12; // rsi
  _QWORD *v13; // r12
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v10 = *a5;
  v15[0] = v10;
  if ( v10 )
    sub_1623A60((__int64)v15, v10, 2);
  v11 = (_QWORD *)sub_145CBF0(*(__int64 **)(a1 + 648), 56, 16);
  v12 = (unsigned __int8 *)v15[0];
  v11[2] = a2;
  v13 = v11;
  v11[3] = a3;
  v11[4] = v12;
  if ( v12 )
    sub_1623210((__int64)v15, v12, (__int64)(v11 + 4));
  *((_DWORD *)v13 + 10) = a6;
  *((_DWORD *)v13 + 11) = 1;
  *((_WORD *)v13 + 24) = 0;
  *v13 = a4;
  return v13;
}
