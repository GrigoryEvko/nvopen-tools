// Function: sub_1D19610
// Address: 0x1d19610
//
_QWORD *__fastcall sub_1D19610(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 *a5)
{
  char v9; // r8
  _QWORD *result; // rax
  __int64 v11; // r9
  int v12; // esi
  __int64 *v13; // r12
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-E8h]
  _QWORD *v17; // [rsp+18h] [rbp-D8h]
  _QWORD *v18; // [rsp+18h] [rbp-D8h]
  _QWORD *v19; // [rsp+18h] [rbp-D8h]
  __int64 v20; // [rsp+20h] [rbp-D0h] BYREF
  int v21; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v22[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v23[176]; // [rsp+40h] [rbp-B0h] BYREF

  v9 = sub_1D12E90(a2);
  result = 0;
  if ( !v9 )
  {
    v11 = *(_QWORD *)(a2 + 40);
    v12 = *(unsigned __int16 *)(a2 + 24);
    v13 = &a3[2 * a4];
    v22[0] = (unsigned __int64)v23;
    v22[1] = 0x2000000000LL;
    v16 = v11;
    sub_16BD430((__int64)v22, v12);
    sub_16BD4C0((__int64)v22, v16);
    while ( v13 != a3 )
    {
      v14 = *a3;
      a3 += 2;
      sub_16BD4C0((__int64)v22, v14);
      sub_16BD430((__int64)v22, *((_DWORD *)a3 - 2));
    }
    sub_1D14B60((__int64)v22, a2);
    v15 = *(_QWORD *)(a2 + 72);
    v20 = v15;
    if ( v15 )
      sub_1623A60((__int64)&v20, v15, 2);
    v21 = *(_DWORD *)(a2 + 64);
    result = sub_1D17920(a1, (__int64)v22, (__int64)&v20, a5);
    if ( v20 )
    {
      v17 = result;
      sub_161E7C0((__int64)&v20, v20);
      result = v17;
    }
    if ( result )
    {
      v18 = result;
      sub_1D19330((__int64)result, *(_WORD *)(a2 + 80));
      result = v18;
    }
    if ( (_BYTE *)v22[0] != v23 )
    {
      v19 = result;
      _libc_free(v22[0]);
      return v19;
    }
  }
  return result;
}
