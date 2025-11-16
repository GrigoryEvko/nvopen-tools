// Function: sub_39A5DF0
// Address: 0x39a5df0
//
__int64 __fastcall sub_39A5DF0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r15
  __int64 v5; // r12
  unsigned __int64 *v6; // rax
  unsigned __int8 *v7; // rsi
  int v8; // r15d
  unsigned __int8 *v9; // rcx
  int v10; // r8d
  int v11; // r9d
  __int64 result; // rax
  unsigned __int8 *v13; // rsi
  unsigned __int8 *v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r8
  _DWORD v17[13]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = sub_39A5D10(a1);
  v5 = sub_145CBF0(a1 + 11, 48, 16);
  *(_QWORD *)v5 = v5 | 4;
  *(_WORD *)(v5 + 28) = 33;
  v6 = *(unsigned __int64 **)(a2 + 32);
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)(v5 + 16) = 0;
  *(_DWORD *)(v5 + 24) = -1;
  *(_BYTE *)(v5 + 30) = 0;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = a2;
  if ( v6 )
  {
    *(_QWORD *)v5 = *v6;
    *v6 = v5 & 0xFFFFFFFFFFFFFFFBLL;
  }
  *(_QWORD *)(a2 + 32) = v5;
  sub_39A3B20((__int64)a1, v5, 73, v4);
  v7 = *(unsigned __int8 **)(a3 - 8LL * *(unsigned int *)(a3 + 8));
  if ( v7 )
  {
    v8 = 1;
    v9 = sub_39A23D0((__int64)a1, v7);
    if ( v9 )
    {
      v8 = 0;
      sub_39A3B20((__int64)a1, v5, 34, (__int64)v9);
    }
  }
  else
  {
    v15 = *(_QWORD *)(a3 + 24);
    v17[0] = 65549;
    v8 = 0;
    sub_39A3860((__int64)a1, (__int64 *)(v5 + 8), 34, (__int64)v17, v15);
  }
  result = 2LL - *(unsigned int *)(a3 + 8);
  v13 = *(unsigned __int8 **)(a3 + 8 * result);
  if ( v13 )
  {
    v14 = sub_39A23D0((__int64)a1, v13);
    if ( !v14 )
    {
      v8 |= 2u;
      return (__int64)sub_39A2960((__int64)a1, a3, v5, v8, v10, v11);
    }
    result = sub_39A3B20((__int64)a1, v5, 47, (__int64)v14);
  }
  else if ( !*(_BYTE *)(a3 + 40) )
  {
    v16 = *(_QWORD *)(a3 + 32);
    v17[0] = 65549;
    result = sub_39A3860((__int64)a1, (__int64 *)(v5 + 8), 47, (__int64)v17, v16);
  }
  if ( v8 )
    return (__int64)sub_39A2960((__int64)a1, a3, v5, v8, v10, v11);
  return result;
}
