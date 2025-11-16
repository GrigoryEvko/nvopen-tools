// Function: sub_234D9C0
// Address: 0x234d9c0
//
__int64 __fastcall sub_234D9C0(__int64 a1, __int64 a2, __int64 a3, char a4, __int64 a5, __int64 a6)
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
  __int64 v24; // [rsp+0h] [rbp-930h]
  __int64 v25; // [rsp+0h] [rbp-930h]
  __int64 v26; // [rsp+0h] [rbp-930h]
  char *v28; // [rsp+10h] [rbp-920h] BYREF
  __int64 v29; // [rsp+18h] [rbp-918h]
  _BYTE v30[2048]; // [rsp+20h] [rbp-910h] BYREF
  __int64 v31; // [rsp+820h] [rbp-110h]
  __int64 v32; // [rsp+828h] [rbp-108h]
  __int64 v33; // [rsp+830h] [rbp-100h]
  unsigned int v34; // [rsp+838h] [rbp-F8h]
  __int64 v35; // [rsp+840h] [rbp-F0h]
  __int64 v36; // [rsp+848h] [rbp-E8h]
  __int64 v37; // [rsp+850h] [rbp-E0h]
  unsigned int v38; // [rsp+858h] [rbp-D8h]
  char *v39; // [rsp+860h] [rbp-D0h] BYREF
  __int64 v40; // [rsp+868h] [rbp-C8h]
  _BYTE v41[128]; // [rsp+870h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+8F0h] [rbp-40h]
  int v43; // [rsp+8F8h] [rbp-38h]

  v6 = a2;
  v8 = a3;
  v9 = *(_DWORD *)(a2 + 8);
  v28 = v30;
  v29 = 0x10000000000LL;
  if ( v9 )
  {
    sub_2303B80((__int64)&v28, (char **)a2, a3, 0x10000000000LL, a5, a6);
    v6 = a2;
  }
  v10 = *(_QWORD *)(v6 + 2072);
  v11 = *(_DWORD *)(v6 + 2136);
  ++*(_QWORD *)(v6 + 2064);
  v32 = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2080);
  ++*(_QWORD *)(v6 + 2096);
  LODWORD(v33) = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2084);
  v31 = 1;
  HIDWORD(v33) = v10;
  LODWORD(v10) = *(_DWORD *)(v6 + 2088);
  *(_QWORD *)(v6 + 2072) = 0;
  v34 = v10;
  v12 = *(_QWORD *)(v6 + 2104);
  *(_QWORD *)(v6 + 2080) = 0;
  v36 = v12;
  LODWORD(v12) = *(_DWORD *)(v6 + 2112);
  *(_DWORD *)(v6 + 2088) = 0;
  LODWORD(v37) = v12;
  LODWORD(v12) = *(_DWORD *)(v6 + 2116);
  v35 = 1;
  HIDWORD(v37) = v12;
  v13 = *(unsigned int *)(v6 + 2120);
  *(_QWORD *)(v6 + 2104) = 0;
  v38 = v13;
  *(_QWORD *)(v6 + 2112) = 0;
  *(_DWORD *)(v6 + 2120) = 0;
  v39 = v41;
  v40 = 0x1000000000LL;
  if ( v11 )
  {
    v24 = v6;
    sub_2303B80((__int64)&v39, (char **)(v6 + 2128), v13, 0x1000000000LL, a5, a6);
    v6 = v24;
  }
  v14 = *(_QWORD *)(v6 + 2272);
  v15 = *(_DWORD *)(v6 + 2280);
  v42 = v14;
  v43 = v15;
  v16 = sub_22077B0(0x8F8u);
  if ( v16 )
  {
    *(_QWORD *)(v16 + 16) = 0x10000000000LL;
    v19 = (unsigned int)v29;
    *(_QWORD *)v16 = &unk_4A11978;
    *(_QWORD *)(v16 + 8) = v16 + 24;
    if ( (_DWORD)v19 )
    {
      v26 = v16;
      sub_2303B80(v16 + 8, &v28, v16 + 24, v19, v17, v18);
      v16 = v26;
    }
    ++v31;
    ++v35;
    *(_QWORD *)(v16 + 2080) = v32;
    v20 = v33;
    *(_QWORD *)(v16 + 2072) = 1;
    *(_QWORD *)(v16 + 2088) = v20;
    v32 = 0;
    *(_DWORD *)(v16 + 2096) = v34;
    v33 = 0;
    *(_QWORD *)(v16 + 2112) = v36;
    v34 = 0;
    *(_QWORD *)(v16 + 2120) = v37;
    LODWORD(v20) = v38;
    *(_QWORD *)(v16 + 2104) = 1;
    *(_DWORD *)(v16 + 2128) = v20;
    *(_QWORD *)(v16 + 2136) = v16 + 2152;
    v21 = (unsigned int)v40;
    v36 = 0;
    v37 = 0;
    v38 = 0;
    *(_QWORD *)(v16 + 2144) = 0x1000000000LL;
    if ( (_DWORD)v21 )
    {
      v25 = v16;
      sub_2303B80(v16 + 2136, &v39, v21, 0x1000000000LL, v17, v18);
      v16 = v25;
    }
    *(_QWORD *)(v16 + 2280) = v42;
    *(_DWORD *)(v16 + 2288) = v43;
  }
  *(_QWORD *)a1 = v16;
  v22 = v39;
  *(_BYTE *)(a1 + 8) = v8;
  *(_BYTE *)(a1 + 9) = a4;
  if ( v22 != v41 )
    _libc_free((unsigned __int64)v22);
  sub_C7D6A0(v36, 8LL * v38, 8);
  sub_C7D6A0(v32, 16LL * v34, 8);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  return a1;
}
