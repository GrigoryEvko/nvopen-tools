// Function: sub_17D3810
// Address: 0x17d3810
//
_QWORD *__fastcall sub_17D3810(__int64 *a1, __int64 a2, _BYTE *a3)
{
  bool v4; // zf
  _QWORD *v5; // r12
  __int64 v6; // rdi
  unsigned __int64 *v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  unsigned __int8 *v14; // [rsp+8h] [rbp-48h] BYREF
  _BYTE *v15; // [rsp+10h] [rbp-40h] BYREF
  __int16 v16; // [rsp+20h] [rbp-30h]

  v4 = *a3 == 0;
  v16 = 257;
  if ( !v4 )
  {
    v15 = a3;
    LOBYTE(v16) = 3;
  }
  v5 = sub_1648A60(64, 1u);
  if ( v5 )
    sub_15F9210((__int64)v5, *(_QWORD *)(*(_QWORD *)a2 + 24LL), a2, 0, 0, 0);
  v6 = a1[1];
  if ( v6 )
  {
    v7 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v6 + 40, (__int64)v5);
    v8 = v5[3];
    v9 = *v7;
    v5[4] = v7;
    v9 &= 0xFFFFFFFFFFFFFFF8LL;
    v5[3] = v9 | v8 & 7;
    *(_QWORD *)(v9 + 8) = v5 + 3;
    *v7 = *v7 & 7 | (unsigned __int64)(v5 + 3);
  }
  sub_164B780((__int64)v5, (__int64 *)&v15);
  v10 = *a1;
  if ( *a1 )
  {
    v14 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v14, v10, 2);
    v11 = v5[6];
    if ( v11 )
      sub_161E7C0((__int64)(v5 + 6), v11);
    v12 = v14;
    v5[6] = v14;
    if ( v12 )
      sub_1623210((__int64)&v14, v12, (__int64)(v5 + 6));
  }
  return v5;
}
