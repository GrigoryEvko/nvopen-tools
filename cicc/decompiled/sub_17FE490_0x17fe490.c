// Function: sub_17FE490
// Address: 0x17fe490
//
__int64 __fastcall sub_17FE490(__int64 *a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 *a6)
{
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rdi
  unsigned __int64 *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rsi
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  const void *v19; // [rsp+8h] [rbp-78h]
  __int64 v21; // [rsp+10h] [rbp-70h]
  unsigned __int8 *v23; // [rsp+28h] [rbp-58h] BYREF
  char v24[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v25; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(a3 + 16) <= 0x10u )
    return sub_15A3A20((__int64 *)a2, (__int64 *)a3, a4, a5, 0);
  v25 = 257;
  v10 = sub_1648A60(88, 2u);
  v11 = v10;
  if ( v10 )
  {
    v19 = a4;
    v21 = (__int64)v10;
    sub_15F1EA0((__int64)v10, *(_QWORD *)a2, 63, (__int64)(v10 - 6), 2, 0);
    v11[7] = v11 + 9;
    v11[8] = 0x400000000LL;
    sub_15FAD90((__int64)v11, a2, a3, v19, a5, (__int64)v24);
  }
  else
  {
    v21 = 0;
  }
  v12 = a1[1];
  if ( v12 )
  {
    v13 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v12 + 40, (__int64)v11);
    v14 = v11[3];
    v15 = *v13;
    v11[4] = v13;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v15 | v14 & 7;
    *(_QWORD *)(v15 + 8) = v11 + 3;
    *v13 = *v13 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780(v21, a6);
  v16 = *a1;
  if ( *a1 )
  {
    v23 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v23, v16, 2);
    v17 = v11[6];
    if ( v17 )
      sub_161E7C0((__int64)(v11 + 6), v17);
    v18 = v23;
    v11[6] = v23;
    if ( v18 )
      sub_1623210((__int64)&v23, v18, (__int64)(v11 + 6));
  }
  return (__int64)v11;
}
