// Function: sub_33ED830
// Address: 0x33ed830
//
_QWORD *__fastcall sub_33ED830(__int64 a1, int a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *result; // rax
  int v10; // eax
  unsigned int v11; // r10d
  __int64 v12; // rax
  __int64 *v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // r9
  __int64 v16; // rsi
  unsigned __int64 v17; // r9
  unsigned __int64 v18; // rsi
  unsigned int v19; // [rsp-58h] [rbp-58h]
  unsigned __int64 v21; // [rsp-50h] [rbp-50h]
  _QWORD *v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-48h] [rbp-48h] BYREF
  int v24; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a5 + 24) != 13 )
    return 0;
  if ( !(*(unsigned __int8 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 16) + 1952LL))(*(_QWORD *)(a1 + 16), a5) )
    return 0;
  v10 = *(_DWORD *)(a6 + 24);
  v11 = a3;
  if ( v10 != 11 && v10 != 35 )
    return 0;
  v12 = *(_QWORD *)(a6 + 96);
  v13 = *(__int64 **)(v12 + 24);
  v14 = *(_DWORD *)(v12 + 32);
  if ( v14 <= 0x40 )
  {
    v15 = 0;
    if ( v14 )
      v15 = (__int64)((_QWORD)v13 << (64 - (unsigned __int8)v14)) >> (64 - (unsigned __int8)v14);
  }
  else
  {
    v15 = *v13;
  }
  if ( a2 != 56 )
  {
    v15 = -v15;
    if ( a2 != 57 )
      return 0;
  }
  v16 = *(_QWORD *)(a6 + 80);
  v17 = *(_QWORD *)(a5 + 104) + v15;
  v23 = v16;
  if ( v16 )
  {
    v19 = a3;
    v21 = v17;
    sub_B96E90((__int64)&v23, v16, 1);
    v11 = v19;
    v17 = v21;
  }
  v18 = *(_QWORD *)(a5 + 96);
  v24 = *(_DWORD *)(a6 + 72);
  result = sub_33ED290(a1, v18, (__int64)&v23, v11, a4, v17, 0, 0);
  if ( v23 )
  {
    v22 = result;
    sub_B91220((__int64)&v23, v23);
    return v22;
  }
  return result;
}
