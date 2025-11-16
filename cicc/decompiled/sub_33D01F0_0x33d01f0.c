// Function: sub_33D01F0
// Address: 0x33d01f0
//
bool __fastcall sub_33D01F0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rbx
  __int64 v14; // [rsp-D0h] [rbp-D0h] BYREF
  unsigned __int16 v15; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v16; // [rsp-C0h] [rbp-C0h]
  _BYTE v17[64]; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD v18[15]; // [rsp-78h] [rbp-78h] BYREF

  if ( (*(_BYTE *)(a2 + 32) & 8) != 0 )
    return 0;
  if ( (*(_BYTE *)(a3 + 32) & 8) != 0 )
    return 0;
  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) != 0 )
    return 0;
  if ( (*(_WORD *)(a2 + 32) & 0x380) != 0 )
    return 0;
  if ( (*(_WORD *)(a3 + 32) & 0x380) != 0 )
    return 0;
  v6 = *(_QWORD *)(a3 + 40);
  v7 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)v7 != *(_QWORD *)v6 )
    return 0;
  if ( *(_DWORD *)(v7 + 8) != *(_DWORD *)(v6 + 8) )
    return 0;
  v9 = *(_QWORD *)(a2 + 104);
  v15 = *(_WORD *)(a2 + 96);
  v16 = v9;
  v18[0] = sub_2D5B750(&v15);
  v18[1] = v10;
  v11 = (unsigned __int64)sub_CA1930(v18) >> 3;
  if ( v11 != a4 )
    return 0;
  sub_33644B0((__int64)v17, a3, a1);
  sub_33644B0((__int64)v18, a2, a1);
  v14 = 0;
  if ( !sub_3364290((__int64)v17, (__int64)v18, a1, &v14) )
    return 0;
  return v11 * a5 == v14;
}
