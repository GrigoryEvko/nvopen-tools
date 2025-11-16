// Function: sub_2353C90
// Address: 0x2353c90
//
void __fastcall sub_2353C90(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // esi
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // [rsp+8h] [rbp-928h] BYREF
  char *v19; // [rsp+10h] [rbp-920h] BYREF
  __int64 v20; // [rsp+18h] [rbp-918h]
  _BYTE v21[2048]; // [rsp+20h] [rbp-910h] BYREF
  __int64 v22; // [rsp+820h] [rbp-110h]
  __int64 v23; // [rsp+828h] [rbp-108h]
  __int64 v24; // [rsp+830h] [rbp-100h]
  unsigned int v25; // [rsp+838h] [rbp-F8h]
  __int64 v26; // [rsp+840h] [rbp-F0h]
  __int64 v27; // [rsp+848h] [rbp-E8h]
  __int64 v28; // [rsp+850h] [rbp-E0h]
  unsigned int v29; // [rsp+858h] [rbp-D8h]
  char *v30; // [rsp+860h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+868h] [rbp-C8h]
  _BYTE v32[128]; // [rsp+870h] [rbp-C0h] BYREF
  __int64 v33; // [rsp+8F0h] [rbp-40h]
  int v34; // [rsp+8F8h] [rbp-38h]

  v7 = *(_DWORD *)(a2 + 8);
  v19 = v21;
  v20 = 0x10000000000LL;
  if ( v7 )
    sub_2303B80((__int64)&v19, (char **)a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a2 + 2072);
  v9 = *(unsigned int *)(a2 + 2136);
  v22 = 1;
  ++*(_QWORD *)(a2 + 2064);
  v23 = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 2080);
  ++*(_QWORD *)(a2 + 2096);
  LODWORD(v24) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 2084);
  *(_QWORD *)(a2 + 2072) = 0;
  HIDWORD(v24) = v8;
  LODWORD(v8) = *(_DWORD *)(a2 + 2088);
  *(_QWORD *)(a2 + 2080) = 0;
  v25 = v8;
  v10 = *(_QWORD *)(a2 + 2104);
  *(_DWORD *)(a2 + 2088) = 0;
  v27 = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 2112);
  v26 = 1;
  LODWORD(v28) = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 2116);
  *(_QWORD *)(a2 + 2104) = 0;
  HIDWORD(v28) = v10;
  LODWORD(v10) = *(_DWORD *)(a2 + 2120);
  *(_QWORD *)(a2 + 2112) = 0;
  v29 = v10;
  *(_DWORD *)(a2 + 2120) = 0;
  v30 = v32;
  v31 = 0x1000000000LL;
  if ( (_DWORD)v9 )
    sub_2303B80((__int64)&v30, (char **)(a2 + 2128), a3, v9, a5, a6);
  v33 = *(_QWORD *)(a2 + 2272);
  v34 = *(_DWORD *)(a2 + 2280);
  v11 = (_QWORD *)sub_22077B0(0x8F8u);
  v15 = (__int64)v11;
  if ( v11 )
  {
    v16 = (unsigned int)v20;
    *v11 = &unk_4A11978;
    v11[1] = v11 + 3;
    v11[2] = 0x10000000000LL;
    if ( (_DWORD)v16 )
      sub_2303B80((__int64)(v11 + 1), &v19, v16, v12, v13, v14);
    ++v22;
    ++v26;
    *(_QWORD *)(v15 + 2080) = v23;
    v17 = v24;
    *(_QWORD *)(v15 + 2072) = 1;
    *(_QWORD *)(v15 + 2088) = v17;
    v23 = 0;
    *(_DWORD *)(v15 + 2096) = v25;
    v24 = 0;
    *(_QWORD *)(v15 + 2112) = v27;
    v25 = 0;
    *(_QWORD *)(v15 + 2120) = v28;
    LODWORD(v17) = v29;
    *(_QWORD *)(v15 + 2104) = 1;
    *(_DWORD *)(v15 + 2128) = v17;
    *(_QWORD *)(v15 + 2136) = v15 + 2152;
    *(_QWORD *)(v15 + 2144) = 0x1000000000LL;
    v27 = 0;
    v28 = 0;
    v29 = 0;
    if ( (_DWORD)v31 )
      sub_2303B80(v15 + 2136, &v30, v16, v12, v13, v14);
    *(_QWORD *)(v15 + 2280) = v33;
    *(_DWORD *)(v15 + 2288) = v34;
  }
  v18 = v15;
  sub_2353900(a1, (unsigned __int64 *)&v18);
  sub_233EFE0(&v18);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  sub_C7D6A0(v27, 8LL * v29, 8);
  sub_C7D6A0(v23, 16LL * v25, 8);
  if ( v19 != v21 )
    _libc_free((unsigned __int64)v19);
}
