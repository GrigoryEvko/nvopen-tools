// Function: sub_2F8EBD0
// Address: 0x2f8ebd0
//
void __fastcall sub_2F8EBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rbx
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  char *v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  char *v25; // rdi
  char *v26; // [rsp+28h] [rbp-108h] BYREF
  __int64 v27; // [rsp+30h] [rbp-100h]
  _BYTE v28[64]; // [rsp+38h] [rbp-F8h] BYREF
  char *v29; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v30; // [rsp+80h] [rbp-B0h]
  _BYTE v31[64]; // [rsp+88h] [rbp-A8h] BYREF
  __int64 v32; // [rsp+C8h] [rbp-68h]
  __int64 v33; // [rsp+D0h] [rbp-60h]
  __int64 v34; // [rsp+D8h] [rbp-58h]
  __int64 v35; // [rsp+E0h] [rbp-50h]
  __int64 v36; // [rsp+E8h] [rbp-48h]
  __int64 v37; // [rsp+F0h] [rbp-40h]
  int v38; // [rsp+F8h] [rbp-38h]
  __int16 v39; // [rsp+FCh] [rbp-34h]
  char v40; // [rsp+FEh] [rbp-32h]

  v7 = *(_QWORD *)(a1 + 48);
  v8 = *(_QWORD *)(a1 + 56);
  if ( v7 != v8 )
  {
    v9 = *(_QWORD *)(a1 + 48);
    do
    {
      v10 = *(_QWORD *)(v9 + 120);
      if ( v10 != v9 + 136 )
        _libc_free(v10);
      v11 = *(_QWORD *)(v9 + 40);
      if ( v11 != v9 + 56 )
        _libc_free(v11);
      v9 += 256;
    }
    while ( v8 != v9 );
    *(_QWORD *)(a1 + 56) = v7;
  }
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  v27 = 0x400000000LL;
  v30 = 0x400000000LL;
  v32 = 0xFFFFFFFFLL;
  v39 = 0;
  v26 = v28;
  v29 = v31;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 0;
  sub_2F8E460(a1 + 112, &v26, 0, a4, a5, a6);
  sub_2F8E460(a1 + 192, &v29, v12, v13, v14, v15);
  v20 = v29;
  *(_QWORD *)(a1 + 272) = v32;
  *(_QWORD *)(a1 + 280) = v33;
  *(_QWORD *)(a1 + 288) = v34;
  *(_QWORD *)(a1 + 296) = v35;
  *(_QWORD *)(a1 + 304) = v36;
  *(_QWORD *)(a1 + 312) = v37;
  *(_DWORD *)(a1 + 320) = v38;
  *(_WORD *)(a1 + 324) = v39;
  *(_BYTE *)(a1 + 326) = v40;
  if ( v20 != v31 )
    _libc_free((unsigned __int64)v20);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 360) = 0;
  v27 = 0x400000000LL;
  v30 = 0x400000000LL;
  v32 = 0xFFFFFFFFLL;
  v39 = 0;
  v26 = v28;
  v29 = v31;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v40 = 0;
  sub_2F8E460(a1 + 368, &v26, v16, v17, v18, v19);
  sub_2F8E460(a1 + 448, &v29, v21, v22, v23, v24);
  v25 = v29;
  *(_QWORD *)(a1 + 528) = v32;
  *(_QWORD *)(a1 + 536) = v33;
  *(_QWORD *)(a1 + 544) = v34;
  *(_QWORD *)(a1 + 552) = v35;
  *(_QWORD *)(a1 + 560) = v36;
  *(_QWORD *)(a1 + 568) = v37;
  *(_DWORD *)(a1 + 576) = v38;
  *(_WORD *)(a1 + 580) = v39;
  *(_BYTE *)(a1 + 582) = v40;
  if ( v25 != v31 )
    _libc_free((unsigned __int64)v25);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
}
