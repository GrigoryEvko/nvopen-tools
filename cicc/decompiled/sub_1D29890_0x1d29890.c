// Function: sub_1D29890
// Address: 0x1d29890
//
__int64 *__fastcall sub_1D29890(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *result; // rax
  int v10; // eax
  unsigned int v11; // r10d
  __int64 v12; // rax
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rsi
  unsigned int v20; // [rsp-58h] [rbp-58h]
  __int64 v22; // [rsp-50h] [rbp-50h]
  __int64 *v23; // [rsp-50h] [rbp-50h]
  __int64 v24; // [rsp-48h] [rbp-48h] BYREF
  int v25; // [rsp-40h] [rbp-40h]

  if ( *(_WORD *)(a5 + 24) != 12 )
    return 0;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 1048LL))(*(_QWORD *)(a1 + 16), a5) )
    return 0;
  v10 = *(unsigned __int16 *)(a6 + 24);
  v11 = a3;
  if ( v10 != 32 && v10 != 10 )
    return 0;
  v12 = *(_QWORD *)(a6 + 88);
  v13 = *(_DWORD *)(v12 + 32);
  v14 = *(__int64 **)(v12 + 24);
  v15 = v13 <= 0x40 ? (__int64)((_QWORD)v14 << (64 - (unsigned __int8)v13)) >> (64 - (unsigned __int8)v13) : *v14;
  if ( a2 != 52 )
  {
    v15 = -v15;
    if ( a2 != 53 )
      return 0;
  }
  v16 = *(_QWORD *)(a6 + 72);
  v17 = *(_QWORD *)(a5 + 96) + v15;
  v18 = v17;
  v24 = v16;
  if ( v16 )
  {
    v20 = a3;
    v22 = v17;
    sub_1623A60((__int64)&v24, v16, 2);
    v11 = v20;
    v18 = v22;
  }
  v19 = *(_QWORD *)(a5 + 88);
  v25 = *(_DWORD *)(a6 + 64);
  result = sub_1D29600((_QWORD *)a1, v19, (__int64)&v24, v11, a4, v18, 0, 0);
  if ( v24 )
  {
    v23 = result;
    sub_161E7C0((__int64)&v24, v24);
    return v23;
  }
  return result;
}
