// Function: sub_1D28D50
// Address: 0x1d28d50
//
__int64 __fastcall sub_1D28D50(_QWORD *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 result; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned __int8 *v16; // rsi
  unsigned __int8 *v17; // [rsp+8h] [rbp-28h] BYREF

  v6 = a2;
  v8 = a1[88];
  v9 = a1[87];
  v10 = (v8 - v9) >> 3;
  if ( a2 >= v10 )
  {
    v12 = (int)(a2 + 1);
    if ( v12 > v10 )
    {
      sub_1D26900((__int64)(a1 + 87), v12 - v10);
      v9 = a1[87];
    }
    else if ( v12 < v10 )
    {
      v13 = v9 + 8 * v12;
      if ( v8 != v13 )
        a1[88] = v13;
    }
  }
  result = *(_QWORD *)(v9 + 8 * v6);
  if ( !result )
  {
    v14 = a1[26];
    if ( v14 )
      a1[26] = *(_QWORD *)v14;
    else
      v14 = sub_145CBF0(a1 + 27, 112, 8);
    v15 = sub_1D274F0(1u, v9, v8, a5, a6);
    v17 = 0;
    *(_QWORD *)v14 = 0;
    v16 = v17;
    *(_QWORD *)(v14 + 40) = v15;
    *(_QWORD *)(v14 + 8) = 0;
    *(_QWORD *)(v14 + 16) = 0;
    *(_WORD *)(v14 + 24) = 7;
    *(_DWORD *)(v14 + 28) = -1;
    *(_QWORD *)(v14 + 32) = 0;
    *(_QWORD *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 56) = 0x100000000LL;
    *(_DWORD *)(v14 + 64) = 0;
    *(_QWORD *)(v14 + 72) = v16;
    if ( v16 )
      sub_1623210((__int64)&v17, v16, v14 + 72);
    *(_DWORD *)(v14 + 84) = a2;
    *(_WORD *)(v14 + 80) &= 0xF000u;
    *(_WORD *)(v14 + 26) = 0;
    *(_QWORD *)(a1[87] + 8 * v6) = v14;
    sub_1D172A0((__int64)a1, v14);
    return *(_QWORD *)(a1[87] + 8 * v6);
  }
  return result;
}
