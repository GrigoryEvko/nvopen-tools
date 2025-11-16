// Function: sub_1AE7E00
// Address: 0x1ae7e00
//
void __fastcall sub_1AE7E00(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 *v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 *v18; // rdx
  __int64 v19; // rsi
  unsigned __int64 v20; // rcx
  __int64 v21; // rcx
  _BYTE *v22; // [rsp+0h] [rbp-80h] BYREF
  __int64 v23; // [rsp+8h] [rbp-78h]
  _BYTE v24[112]; // [rsp+10h] [rbp-70h] BYREF

  v23 = 0x800000000LL;
  v22 = v24;
  sub_15B13F0((__int64)&v22, a3);
  v4 = *a1;
  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 24 * (2 - v5)) + 24LL);
  if ( (_DWORD)v23 )
  {
    v7 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v7 + 16) )
      BUG();
    v6 = sub_15C46E0(*(_QWORD **)(*(_QWORD *)(a2 + 24 * (2 - v5)) + 24LL), (__int64)&v22, *(_DWORD *)(v7 + 36) == 38);
  }
  v8 = *(_QWORD *)(v4 + 8);
  if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
    v9 = *(__int64 **)(v8 - 8);
  else
    v9 = (__int64 *)(v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF));
  v10 = **(__int64 ***)v4;
  v11 = sub_1624210(*v9);
  v12 = sub_1628DA0(v10, (__int64)v11);
  v13 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *v13 )
  {
    v14 = v13[1];
    v15 = v13[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v15 = v14;
    if ( v14 )
      *(_QWORD *)(v14 + 16) = *(_QWORD *)(v14 + 16) & 3LL | v15;
  }
  *v13 = v12;
  if ( v12 )
  {
    v16 = *(_QWORD *)(v12 + 8);
    v13[1] = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = (unsigned __int64)(v13 + 1) | *(_QWORD *)(v16 + 16) & 3LL;
    v13[2] = (v12 + 8) | v13[2] & 3;
    *(_QWORD *)(v12 + 8) = v13;
  }
  v17 = sub_1628DA0(*(__int64 **)(v4 + 16), v6);
  v18 = (__int64 *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *v18 )
  {
    v19 = v18[1];
    v20 = v18[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v20 = v19;
    if ( v19 )
      *(_QWORD *)(v19 + 16) = *(_QWORD *)(v19 + 16) & 3LL | v20;
  }
  *v18 = v17;
  if ( v17 )
  {
    v21 = *(_QWORD *)(v17 + 8);
    v18[1] = v21;
    if ( v21 )
      *(_QWORD *)(v21 + 16) = (unsigned __int64)(v18 + 1) | *(_QWORD *)(v21 + 16) & 3LL;
    v18[2] = (v17 + 8) | v18[2] & 3;
    *(_QWORD *)(v17 + 8) = v18;
  }
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
}
