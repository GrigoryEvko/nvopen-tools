// Function: sub_254B630
// Address: 0x254b630
//
__int64 __fastcall sub_254B630(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int64 v5; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-88h]
  unsigned __int64 v7; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-78h]
  _QWORD v9[14]; // [rsp+20h] [rbp-70h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v9[5] = 0x100000000LL;
  v9[6] = a1;
  v9[0] = &unk_49DD210;
  memset(&v9[1], 0, 32);
  sub_CB5980((__int64)v9, 0, 0, 0);
  v2 = sub_904010((__int64)v9, "range(");
  v3 = sub_CB59D0(v2, *(unsigned int *)(a2 + 96));
  sub_904010(v3, ")<");
  v6 = *(_DWORD *)(a2 + 144);
  if ( v6 > 0x40 )
    sub_C43780((__int64)&v5, (const void **)(a2 + 136));
  else
    v5 = *(_QWORD *)(a2 + 136);
  v8 = *(_DWORD *)(a2 + 160);
  if ( v8 > 0x40 )
    sub_C43780((__int64)&v7, (const void **)(a2 + 152));
  else
    v7 = *(_QWORD *)(a2 + 152);
  sub_ABE8C0((__int64)&v5, (__int64)v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  sub_904010((__int64)v9, " / ");
  v6 = *(_DWORD *)(a2 + 112);
  if ( v6 > 0x40 )
    sub_C43780((__int64)&v5, (const void **)(a2 + 104));
  else
    v5 = *(_QWORD *)(a2 + 104);
  v8 = *(_DWORD *)(a2 + 128);
  if ( v8 > 0x40 )
    sub_C43780((__int64)&v7, (const void **)(a2 + 120));
  else
    v7 = *(_QWORD *)(a2 + 120);
  sub_ABE8C0((__int64)&v5, (__int64)v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0(v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0(v5);
  sub_904010((__int64)v9, ">");
  v9[0] = &unk_49DD210;
  sub_CB5840((__int64)v9);
  return a1;
}
