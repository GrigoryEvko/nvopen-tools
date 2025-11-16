// Function: sub_394FCE0
// Address: 0x394fce0
//
__int64 __fastcall sub_394FCE0(__int64 a1, __int64 *a2)
{
  _QWORD *v4; // rdi
  __int64 **v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  unsigned __int8 *v18; // [rsp+8h] [rbp-78h] BYREF
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  __int16 v20; // [rsp+20h] [rbp-60h]
  char v21[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v22; // [rsp+40h] [rbp-40h]

  v4 = (_QWORD *)a2[3];
  v20 = 257;
  v5 = (__int64 **)sub_1643330(v4);
  v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v7 = *(_QWORD *)(a1 + 24 * (1 - v6));
  if ( v5 != *(__int64 ***)v7 )
  {
    if ( *(_BYTE *)(v7 + 16) > 0x10u )
    {
      v9 = *(_QWORD **)(a1 + 24 * (1 - v6));
      v22 = 257;
      v10 = sub_15FE0A0(v9, (__int64)v5, 0, (__int64)v21, 0);
      v11 = a2[1];
      v7 = v10;
      if ( v11 )
      {
        v12 = (__int64 *)a2[2];
        sub_157E9D0(v11 + 40, v10);
        v13 = *(_QWORD *)(v7 + 24);
        v14 = *v12;
        *(_QWORD *)(v7 + 32) = v12;
        v14 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v7 + 24) = v14 | v13 & 7;
        *(_QWORD *)(v14 + 8) = v7 + 24;
        *v12 = *v12 & 7 | (v7 + 24);
      }
      sub_164B780(v7, &v19);
      v15 = *a2;
      if ( *a2 )
      {
        v18 = (unsigned __int8 *)*a2;
        sub_1623A60((__int64)&v18, v15, 2);
        v16 = *(_QWORD *)(v7 + 48);
        if ( v16 )
          sub_161E7C0(v7 + 48, v16);
        v17 = v18;
        *(_QWORD *)(v7 + 48) = v18;
        if ( v17 )
          sub_1623210((__int64)&v18, v17, v7 + 48);
      }
      v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
    else
    {
      v7 = sub_15A4750(*(__int64 ****)(a1 + 24 * (1 - v6)), v5, 0);
      v6 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    }
  }
  sub_15E7280(a2, *(_QWORD **)(a1 - 24 * v6), v7, *(__int64 **)(a1 + 24 * (2 - v6)), 1u, 0, 0, 0, 0);
  return *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
}
