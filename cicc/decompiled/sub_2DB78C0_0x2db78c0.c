// Function: sub_2DB78C0
// Address: 0x2db78c0
//
__int64 __fastcall sub_2DB78C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 *v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // r12d
  __m128i v18; // [rsp+0h] [rbp-3A0h] BYREF
  __int64 v19; // [rsp+60h] [rbp-340h]
  __int64 v20; // [rsp+68h] [rbp-338h]
  __int64 v21; // [rsp+70h] [rbp-330h]
  __int64 v22; // [rsp+78h] [rbp-328h]
  __int64 v23; // [rsp+80h] [rbp-320h]
  _BYTE *v24; // [rsp+C0h] [rbp-2E0h]
  __int64 v25; // [rsp+C8h] [rbp-2D8h]
  _BYTE v26[256]; // [rsp+D0h] [rbp-2D0h] BYREF
  _BYTE *v27; // [rsp+1D0h] [rbp-1D0h]
  __int64 v28; // [rsp+1D8h] [rbp-1C8h]
  _BYTE v29[160]; // [rsp+1E0h] [rbp-1C0h] BYREF
  __int64 v30; // [rsp+280h] [rbp-120h]
  char *v31; // [rsp+288h] [rbp-118h]
  __int64 v32; // [rsp+290h] [rbp-110h]
  int v33; // [rsp+298h] [rbp-108h]
  char v34; // [rsp+29Ch] [rbp-104h]
  char v35; // [rsp+2A0h] [rbp-100h] BYREF
  _BYTE *v36; // [rsp+2E0h] [rbp-C0h]
  __int64 v37; // [rsp+2E8h] [rbp-B8h]
  _BYTE v38[48]; // [rsp+2F0h] [rbp-B0h] BYREF
  int v39; // [rsp+320h] [rbp-80h]
  _BYTE *v40; // [rsp+328h] [rbp-78h]
  __int64 v41; // [rsp+330h] [rbp-70h]
  _BYTE v42[32]; // [rsp+338h] [rbp-68h] BYREF
  unsigned __int64 v43; // [rsp+358h] [rbp-48h]
  int v44; // [rsp+360h] [rbp-40h]
  __int64 v45; // [rsp+368h] [rbp-38h]

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_33:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_501FE44 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_33;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_501FE44);
  v6 = *(__int64 **)(a1 + 8);
  v7 = v5 + 200;
  v8 = *v6;
  v9 = v6[1];
  if ( v8 == v9 )
LABEL_31:
    BUG();
  while ( *(_UNKNOWN **)v8 != &unk_50208AC )
  {
    v8 += 16;
    if ( v9 == v8 )
      goto LABEL_31;
  }
  v10 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v8 + 8) + 104LL))(*(_QWORD *)(v8 + 8), &unk_50208AC);
  v11 = *(__int64 **)(a1 + 8);
  v12 = v10 + 200;
  v13 = *v11;
  v14 = v11[1];
  if ( v13 == v14 )
LABEL_32:
    BUG();
  while ( *(_UNKNOWN **)v13 != &unk_502234C )
  {
    v13 += 16;
    if ( v14 == v13 )
      goto LABEL_32;
  }
  v15 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v13 + 8) + 104LL))(*(_QWORD *)(v13 + 8), &unk_502234C);
  v20 = v7;
  v21 = v12;
  v22 = v15 + 200;
  v28 = 0x400000000LL;
  v18 = 0u;
  v19 = 0;
  v23 = 0;
  v24 = v26;
  v25 = 0x800000000LL;
  v27 = v29;
  v30 = 0;
  v31 = &v35;
  v32 = 8;
  v33 = 0;
  v34 = 1;
  v36 = v38;
  v37 = 0x600000000LL;
  v39 = 0;
  v40 = v42;
  v41 = 0x800000000LL;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v16 = sub_2DB5D20(&v18, a2);
  if ( v43 )
    _libc_free(v43);
  if ( v40 != v42 )
    _libc_free((unsigned __int64)v40);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
  if ( !v34 )
    _libc_free((unsigned __int64)v31);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  if ( v24 != v26 )
    _libc_free((unsigned __int64)v24);
  return v16;
}
