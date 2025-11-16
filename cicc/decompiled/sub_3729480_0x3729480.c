// Function: sub_3729480
// Address: 0x3729480
//
__int64 __fastcall sub_3729480(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // r8
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned __int8 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  unsigned __int8 *v17; // rax
  unsigned __int64 v19; // r12
  const void *v20; // r10
  _BYTE *v21; // rax
  unsigned __int8 *v22; // rax
  size_t v23; // rdx
  _QWORD *v24; // rdi
  _BYTE *v25; // rdi
  const void *v26; // [rsp+8h] [rbp-138h]
  size_t v27; // [rsp+10h] [rbp-130h]
  __int64 v28[3]; // [rsp+30h] [rbp-110h] BYREF
  unsigned __int64 v29; // [rsp+48h] [rbp-F8h]
  void *dest; // [rsp+50h] [rbp-F0h]
  __int64 v31; // [rsp+58h] [rbp-E8h]
  _BYTE **v32; // [rsp+60h] [rbp-E0h]
  _BYTE *v33; // [rsp+70h] [rbp-D0h] BYREF
  unsigned __int64 v34; // [rsp+78h] [rbp-C8h]
  __int64 v35; // [rsp+80h] [rbp-C0h]
  _BYTE v36[184]; // [rsp+88h] [rbp-B8h] BYREF

  v7 = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 64);
  v8 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(*(_QWORD *)(v8 + 200) + 876LL) & 4) == 0 )
    goto LABEL_2;
  v19 = *(_QWORD *)(v7 + 136);
  v34 = 0;
  v20 = *(const void **)(v7 + 128);
  v33 = v36;
  v35 = 128;
  if ( v19 > 0x80 )
  {
    v26 = v20;
    sub_C8D290((__int64)&v33, v36, v19, 1u, v6, (__int64)&v33);
    v20 = v26;
    v25 = &v33[v34];
    goto LABEL_16;
  }
  if ( v19 )
  {
    v25 = v36;
LABEL_16:
    memcpy(v25, v20, v19);
    v19 += v34;
  }
  v31 = 0x100000000LL;
  v34 = v19;
  v28[1] = 2;
  v28[0] = (__int64)&unk_49DD288;
  v28[2] = 0;
  v29 = 0;
  dest = 0;
  v32 = &v33;
  sub_CB5980((__int64)v28, 0, 0, 0);
  v21 = dest;
  if ( (unsigned __int64)dest >= v29 )
  {
    sub_CB5D20((__int64)v28, 46);
  }
  else
  {
    dest = (char *)dest + 1;
    *v21 = 46;
  }
  v22 = (unsigned __int8 *)sub_BD5D20(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 232LL));
  if ( v23 > v29 - (unsigned __int64)dest )
  {
    sub_CB6200((__int64)v28, v22, v23);
  }
  else if ( v23 )
  {
    v27 = v23;
    memcpy(dest, v22, v23);
    dest = (char *)dest + v27;
  }
  v28[0] = (__int64)&unk_49DD388;
  sub_CB5840((__int64)v28);
  v24 = *(_QWORD **)(*(_QWORD *)(a1 + 8) + 216LL);
  BYTE4(v28[0]) = 0;
  v7 = sub_E6E320(v24, v33, v34, *(_BYTE *)(v7 + 188), *(_WORD *)(v7 + 148), 0, v28[0]);
  if ( v33 != v36 )
    _libc_free((unsigned __int64)v33);
  v8 = *(_QWORD *)(a1 + 8);
LABEL_2:
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(v8 + 224) + 176LL))(*(_QWORD *)(v8 + 224), v7, 0);
  v9 = sub_37F9090(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 232LL));
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 208LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
    v9,
    0);
  sub_31DCA10(*(_QWORD *)(a1 + 8), 0);
  v10 = 0xFFFFFFFFLL;
  LODWORD(v11) = sub_AE4380(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 2488LL) + 312LL, 0);
  if ( (_DWORD)v11 )
  {
    _BitScanReverse64((unsigned __int64 *)&v11, (unsigned int)v11);
    v10 = 63 - ((unsigned int)v11 ^ 0x3F);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 608LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
    v10,
    0,
    1,
    0);
  v12 = *(_QWORD *)(a1 + 8);
  v13 = *(_QWORD *)(v12 + 224);
  v14 = (unsigned __int8 *)sub_E808D0(a2, 0, *(_QWORD **)(v12 + 216), 0);
  sub_E9A5B0(v13, v14);
  v15 = *(_QWORD *)(a1 + 8);
  v16 = *(_QWORD *)(v15 + 224);
  v17 = (unsigned __int8 *)sub_E808D0(a3, 0, *(_QWORD **)(v15 + 216), 0);
  return sub_E9A5B0(v16, v17);
}
