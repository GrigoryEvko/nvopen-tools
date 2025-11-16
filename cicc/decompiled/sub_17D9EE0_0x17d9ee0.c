// Function: sub_17D9EE0
// Address: 0x17d9ee0
//
unsigned __int64 __fastcall sub_17D9EE0(__int128 a1)
{
  __int64 v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 *v5; // rbx
  _QWORD *v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r14
  _QWORD *v9; // rax
  __int64 *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 result; // rax
  __int64 *v14; // [rsp+0h] [rbp-D0h]
  __int64 v15; // [rsp+8h] [rbp-C8h]
  const char *v16; // [rsp+10h] [rbp-C0h] BYREF
  char v17; // [rsp+20h] [rbp-B0h]
  char v18; // [rsp+21h] [rbp-AFh]
  char v19[16]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v20; // [rsp+40h] [rbp-90h]
  __int64 v21; // [rsp+50h] [rbp-80h] BYREF
  __int64 v22; // [rsp+58h] [rbp-78h]
  __int64 *v23; // [rsp+60h] [rbp-70h]

  v1 = *((_QWORD *)&a1 + 1);
  v2 = *((_QWORD *)&a1 + 1);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL);
  sub_17D5820(a1, v2);
  sub_17CE510((__int64)&v21, v1, 0, 0, 0);
  v18 = 1;
  v16 = "_msprop";
  v3 = *(_QWORD *)(v1 - 24);
  v17 = 3;
  v15 = v3;
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(v1 - 8);
  else
    v4 = v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v4 + 24);
  v5 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v6 = *(_QWORD **)(v1 - 8);
  else
    v6 = (_QWORD *)(v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v6;
  v7 = sub_17D4DA0(a1);
  if ( *((_BYTE *)v7 + 16) > 0x10u || *((_BYTE *)v5 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
  {
    v14 = v7;
    v20 = 257;
    v9 = sub_1648A60(56, 3u);
    v8 = (__int64)v9;
    if ( v9 )
      sub_15FA660((__int64)v9, v14, (__int64)v5, (_QWORD *)v15, (__int64)v19, 0);
    if ( v22 )
    {
      v10 = v23;
      sub_157E9D0(v22 + 40, v8);
      v11 = *(_QWORD *)(v8 + 24);
      v12 = *v10;
      *(_QWORD *)(v8 + 32) = v10;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v8 + 24) = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v8 + 24;
      *v10 = *v10 & 7 | (v8 + 24);
    }
    sub_164B780(v8, (__int64 *)&v16);
    sub_12A86E0(&v21, v8);
  }
  else
  {
    v8 = sub_15A3950((__int64)v7, (__int64)v5, (_BYTE *)v15, 0);
  }
  sub_17D4920(a1, (__int64 *)v1, v8);
  result = *(unsigned int *)(*(_QWORD *)(a1 + 8) + 156LL);
  if ( (_DWORD)result )
    result = sub_17D9C10((_QWORD *)a1, v1);
  if ( v21 )
    return sub_161E7C0((__int64)&v21, v21);
  return result;
}
