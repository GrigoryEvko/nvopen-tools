// Function: sub_B28E70
// Address: 0xb28e70
//
__int64 __fastcall sub_B28E70(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 *v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // rsi
  __int64 v9; // rbx
  __int64 **v10; // r14
  unsigned int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rsi
  _BYTE *v14; // rbx
  __int64 result; // rax
  _BYTE *v16; // r12
  _BYTE *v17; // rdi
  char *v19[2]; // [rsp+20h] [rbp-1090h] BYREF
  char v20; // [rsp+30h] [rbp-1080h] BYREF
  _QWORD v21[2]; // [rsp+50h] [rbp-1060h] BYREF
  _QWORD v22[64]; // [rsp+60h] [rbp-1050h] BYREF
  _BYTE *v23; // [rsp+260h] [rbp-E50h]
  __int64 v24; // [rsp+268h] [rbp-E48h]
  _BYTE v25[3584]; // [rsp+270h] [rbp-E40h] BYREF
  __int64 v26; // [rsp+1070h] [rbp-40h]

  v2 = 0;
  v4 = (__int64 *)(a1 + 48);
  v5 = v4[10];
  sub_B1AD90(v4, a2);
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_BYTE *)(a1 + 136) = 0;
  *(_DWORD *)(a1 + 140) = 0;
  *(_QWORD *)(a1 + 128) = v5;
  if ( a2 )
  {
    v2 = *(_QWORD *)(a2 + 16);
    if ( v2 )
    {
      v6 = *(_QWORD *)(a2 + 8);
      if ( v2 != v6 )
        sub_B1C1A0(*(_QWORD *)(a2 + 8), *(_QWORD *)(a2 + 16));
      if ( v2 + 304 != v6 + 304 )
        sub_B1C1A0(v6 + 304, v2 + 304);
      *(_BYTE *)(v6 + 608) = *(_BYTE *)(v2 + 608);
      sub_B18600(v6 + 616, v2 + 616);
      v2 = a2;
    }
  }
  v21[0] = v22;
  v21[1] = 0x4000000001LL;
  v23 = v25;
  v24 = 0x4000000000LL;
  v22[0] = 0;
  v26 = v2;
  sub_B28710(v19, a1, v2);
  sub_B187A0(a1, v19);
  if ( v19[0] != &v20 )
    _libc_free(v19[0], v19);
  v7 = sub_B20CA0((__int64)v21, 0);
  v8 = 0;
  *(_QWORD *)(v7 + 8) = 0x100000001LL;
  *(_DWORD *)v7 = 1;
  sub_B1A4E0((__int64)v21, 0);
  v9 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = *(__int64 ***)a1;
    v11 = 1;
    do
    {
      v8 = *v10++;
      v11 = sub_B27790((__int64)v21, v8, v11, (unsigned __int8 (__fastcall *)(__int64, __int64))sub_B184A0, 1u, 0);
    }
    while ( (__int64 **)v9 != v10 );
  }
  sub_B20E50((__int64)v21);
  if ( a2 )
    *(_BYTE *)a2 = 1;
  if ( !*(_DWORD *)(a1 + 8) )
    return sub_B1ACF0((__int64)v21, (__int64)v8);
  v12 = (__int64 *)sub_B1BBB0(a1, 0, 0);
  v13 = a1;
  *(_QWORD *)(a1 + 120) = v12;
  sub_B214B0(v21, a1, *v12);
  v14 = v23;
  result = 7LL * (unsigned int)v24;
  v16 = &v23[56 * (unsigned int)v24];
  if ( v23 != v16 )
  {
    do
    {
      v16 -= 56;
      v17 = (_BYTE *)*((_QWORD *)v16 + 3);
      result = (__int64)(v16 + 40);
      if ( v17 != v16 + 40 )
        result = _libc_free(v17, v13);
    }
    while ( v14 != v16 );
    v16 = v23;
  }
  if ( v16 != v25 )
    result = _libc_free(v16, v13);
  if ( (_QWORD *)v21[0] != v22 )
    return _libc_free(v21[0], v13);
  return result;
}
