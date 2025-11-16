// Function: sub_234C210
// Address: 0x234c210
//
__int64 __fastcall sub_234C210(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  char v8; // bl
  int v9; // edi
  __int64 v10; // rdx
  int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rdx
  char *v22; // rdi
  __int64 v24; // [rsp+8h] [rbp-928h]
  __int64 v25; // [rsp+8h] [rbp-928h]
  __int64 v26; // [rsp+8h] [rbp-928h]
  char *v27; // [rsp+10h] [rbp-920h] BYREF
  __int64 v28; // [rsp+18h] [rbp-918h]
  _BYTE v29[2048]; // [rsp+20h] [rbp-910h] BYREF
  __int64 v30; // [rsp+820h] [rbp-110h]
  __int64 v31; // [rsp+828h] [rbp-108h]
  __int64 v32; // [rsp+830h] [rbp-100h]
  unsigned int v33; // [rsp+838h] [rbp-F8h]
  __int64 v34; // [rsp+840h] [rbp-F0h]
  __int64 v35; // [rsp+848h] [rbp-E8h]
  __int64 v36; // [rsp+850h] [rbp-E0h]
  unsigned int v37; // [rsp+858h] [rbp-D8h]
  char *v38; // [rsp+860h] [rbp-D0h] BYREF
  __int64 v39; // [rsp+868h] [rbp-C8h]
  _BYTE v40[128]; // [rsp+870h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+8F0h] [rbp-40h]
  int v42; // [rsp+8F8h] [rbp-38h]

  v6 = a2;
  v8 = a3;
  v9 = *(_DWORD *)(a2 + 8);
  v27 = v29;
  v28 = 0x10000000000LL;
  if ( v9 )
  {
    sub_2303B80((__int64)&v27, (char **)a2, a3, 0x10000000000LL, a5, a6);
    v6 = a2;
  }
  v10 = *(_QWORD *)(v6 + 2072);
  v11 = *(_DWORD *)(v6 + 2136);
  ++*(_QWORD *)(v6 + 2064);
  v31 = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2080);
  ++*(_QWORD *)(v6 + 2096);
  LODWORD(v32) = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2084);
  v30 = 1;
  HIDWORD(v32) = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2088);
  *(_QWORD *)(v6 + 2072) = 0;
  v33 = v10;
  v12 = *(_QWORD *)(v6 + 2104);
  *(_QWORD *)(v6 + 2080) = 0;
  v35 = v12;
  LODWORD(v12) = *(_DWORD *)(v6 + 2112);
  *(_DWORD *)(v6 + 2088) = 0;
  LODWORD(v36) = v12;
  LODWORD(v12) = *(_DWORD *)(v6 + 2116);
  v34 = 1;
  HIDWORD(v36) = v12;
  v13 = *(unsigned int *)(v6 + 2120);
  *(_QWORD *)(v6 + 2104) = 0;
  v37 = v13;
  *(_QWORD *)(v6 + 2112) = 0;
  *(_DWORD *)(v6 + 2120) = 0;
  v38 = v40;
  v39 = 0x1000000000LL;
  if ( v11 )
  {
    v24 = v6;
    sub_2303B80((__int64)&v38, (char **)(v6 + 2128), v13, 0x1000000000LL, a5, a6);
    v6 = v24;
  }
  v14 = *(_QWORD *)(v6 + 2272);
  v15 = *(_DWORD *)(v6 + 2280);
  v41 = v14;
  v42 = v15;
  v16 = sub_22077B0(0x8F8u);
  if ( v16 )
  {
    *(_QWORD *)(v16 + 16) = 0x10000000000LL;
    v19 = (unsigned int)v28;
    *(_QWORD *)v16 = &unk_4A11978;
    *(_QWORD *)(v16 + 8) = v16 + 24;
    if ( (_DWORD)v19 )
    {
      v26 = v16;
      sub_2303B80(v16 + 8, &v27, v16 + 24, v19, v17, v18);
      v16 = v26;
    }
    ++v30;
    ++v34;
    *(_QWORD *)(v16 + 2080) = v31;
    v20 = v32;
    *(_QWORD *)(v16 + 2072) = 1;
    *(_QWORD *)(v16 + 2088) = v20;
    v31 = 0;
    *(_DWORD *)(v16 + 2096) = v33;
    v32 = 0;
    *(_QWORD *)(v16 + 2112) = v35;
    v33 = 0;
    *(_QWORD *)(v16 + 2120) = v36;
    LODWORD(v20) = v37;
    *(_QWORD *)(v16 + 2104) = 1;
    *(_DWORD *)(v16 + 2128) = v20;
    *(_QWORD *)(v16 + 2136) = v16 + 2152;
    v21 = (unsigned int)v39;
    v35 = 0;
    v36 = 0;
    v37 = 0;
    *(_QWORD *)(v16 + 2144) = 0x1000000000LL;
    if ( (_DWORD)v21 )
    {
      v25 = v16;
      sub_2303B80(v16 + 2136, &v38, v21, 0x1000000000LL, v17, v18);
      v16 = v25;
    }
    *(_QWORD *)(v16 + 2280) = v41;
    *(_DWORD *)(v16 + 2288) = v42;
  }
  v22 = v38;
  *(_QWORD *)a1 = v16;
  *(_BYTE *)(a1 + 8) = v8;
  if ( v22 != v40 )
    _libc_free((unsigned __int64)v22);
  sub_C7D6A0(v35, 8LL * v37, 8);
  sub_C7D6A0(v31, 16LL * v33, 8);
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return a1;
}
