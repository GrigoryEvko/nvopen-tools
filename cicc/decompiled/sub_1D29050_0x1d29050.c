// Function: sub_1D29050
// Address: 0x1d29050
//
__int64 __fastcall sub_1D29050(__int64 a1, int a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6, unsigned int a7)
{
  __int128 v8; // rdi
  int v10; // r12d
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r15
  unsigned __int8 *v15; // rsi
  _QWORD v17[7]; // [rsp+8h] [rbp-38h] BYREF

  *((_QWORD *)&v8 + 1) = a5;
  v10 = a6;
  *(_QWORD *)&v8 = (unsigned int)a4;
  v12 = sub_1D274F0(v8, (__int64)a3, a4, a5, a6);
  v13 = *a3;
  v14 = v12;
  v17[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v17, v13, 2);
    v15 = (unsigned __int8 *)v17[0];
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_WORD *)(a1 + 24) = 159;
    *(_DWORD *)(a1 + 28) = -1;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = v14;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0x100000000LL;
    *(_DWORD *)(a1 + 64) = a2;
    *(_QWORD *)(a1 + 72) = v15;
    if ( v15 )
      sub_1623210((__int64)v17, v15, a1 + 72);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_WORD *)(a1 + 24) = 159;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_DWORD *)(a1 + 28) = -1;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 40) = v12;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = 0x100000000LL;
    *(_DWORD *)(a1 + 64) = a2;
    *(_QWORD *)(a1 + 72) = 0;
  }
  *(_DWORD *)(a1 + 84) = v10;
  *(_WORD *)(a1 + 80) &= 0xF000u;
  *(_WORD *)(a1 + 26) = 0;
  *(_DWORD *)(a1 + 88) = a7;
  return a7;
}
