// Function: sub_1F013F0
// Address: 0x1f013f0
//
void __fastcall sub_1F013F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rcx
  int v17; // r8d
  int v18; // r9d
  char v19; // dl
  char *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // rcx
  int v24; // r8d
  int v25; // r9d
  char v26; // dl
  char *v27; // rdi
  char *v28; // [rsp+20h] [rbp-120h] BYREF
  __int64 v29; // [rsp+28h] [rbp-118h]
  _BYTE v30[64]; // [rsp+30h] [rbp-110h] BYREF
  char *v31; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v32; // [rsp+78h] [rbp-C8h]
  _BYTE v33[64]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+C0h] [rbp-80h]
  __int64 v35; // [rsp+C8h] [rbp-78h]
  __int64 v36; // [rsp+D0h] [rbp-70h]
  __int64 v37; // [rsp+D8h] [rbp-68h]
  int v38; // [rsp+E0h] [rbp-60h]
  __int16 v39; // [rsp+E4h] [rbp-5Ch]
  int v40; // [rsp+E8h] [rbp-58h]
  char v41; // [rsp+ECh] [rbp-54h]
  __int64 v42; // [rsp+F0h] [rbp-50h]
  __int64 v43; // [rsp+F8h] [rbp-48h]
  __int64 v44; // [rsp+100h] [rbp-40h]
  __int64 v45; // [rsp+108h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 56);
  if ( v7 != v8 )
  {
    v9 = *(_QWORD *)(a1 + 48);
    do
    {
      v10 = *(_QWORD *)(v9 + 112);
      if ( v10 != v9 + 128 )
        _libc_free(v10);
      v11 = *(_QWORD *)(v9 + 32);
      if ( v11 != v9 + 48 )
        _libc_free(v11);
      v9 += 272;
    }
    while ( v8 != v9 );
    *(_QWORD *)(a1 + 56) = v7;
  }
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  v29 = 0x400000000LL;
  v32 = 0x400000000LL;
  v34 = 0xFFFFFFFFLL;
  v39 = 0;
  v41 &= 0xFCu;
  v28 = v30;
  v31 = v33;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  sub_1F00E30(a1 + 104, &v28, 0, a4, a5, a6);
  sub_1F00E30(a1 + 184, &v31, v12, v13, v14, v15);
  v19 = v41;
  v20 = v31;
  *(_QWORD *)(a1 + 264) = v34;
  v21 = v19 & 3;
  *(_QWORD *)(a1 + 272) = v35;
  *(_QWORD *)(a1 + 280) = v36;
  *(_QWORD *)(a1 + 288) = v37;
  *(_DWORD *)(a1 + 296) = v38;
  *(_WORD *)(a1 + 300) = v39;
  *(_DWORD *)(a1 + 304) = v40;
  *(_BYTE *)(a1 + 308) = v21 | *(_BYTE *)(a1 + 308) & 0xFC;
  *(_QWORD *)(a1 + 312) = v42;
  *(_QWORD *)(a1 + 320) = v43;
  *(_QWORD *)(a1 + 328) = v44;
  *(_QWORD *)(a1 + 336) = v45;
  if ( v20 != v33 )
    _libc_free((unsigned __int64)v20);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  *(_QWORD *)(a1 + 368) = 0;
  v29 = 0x400000000LL;
  v32 = 0x400000000LL;
  v34 = 0xFFFFFFFFLL;
  v39 = 0;
  v41 &= 0xFCu;
  v28 = v30;
  v31 = v33;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  sub_1F00E30(a1 + 376, &v28, v21, v16, v17, v18);
  sub_1F00E30(a1 + 456, &v31, v22, v23, v24, v25);
  v26 = v41;
  v27 = v31;
  *(_QWORD *)(a1 + 536) = v34;
  *(_QWORD *)(a1 + 544) = v35;
  *(_QWORD *)(a1 + 552) = v36;
  *(_QWORD *)(a1 + 560) = v37;
  *(_DWORD *)(a1 + 568) = v38;
  *(_WORD *)(a1 + 572) = v39;
  *(_DWORD *)(a1 + 576) = v40;
  *(_BYTE *)(a1 + 580) = v26 & 3 | *(_BYTE *)(a1 + 580) & 0xFC;
  *(_QWORD *)(a1 + 584) = v42;
  *(_QWORD *)(a1 + 592) = v43;
  *(_QWORD *)(a1 + 600) = v44;
  *(_QWORD *)(a1 + 608) = v45;
  if ( v27 != v33 )
    _libc_free((unsigned __int64)v27);
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
}
