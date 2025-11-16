// Function: sub_2C28C90
// Address: 0x2c28c90
//
__int64 __fastcall sub_2C28C90(
        _QWORD *a1,
        int a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 *a8,
        _QWORD *a9)
{
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r13
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 *v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // rcx
  int v28; // r14d
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r13
  __int64 *v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v36; // [rsp+0h] [rbp-D0h]
  __int64 v39; // [rsp+10h] [rbp-C0h]
  __int64 v41; // [rsp+20h] [rbp-B0h]
  int v43; // [rsp+3Ch] [rbp-94h] BYREF
  __int64 v44; // [rsp+40h] [rbp-90h] BYREF
  __int64 v45; // [rsp+48h] [rbp-88h] BYREF
  __int64 v46; // [rsp+50h] [rbp-80h] BYREF
  __int64 v47; // [rsp+58h] [rbp-78h]
  __int64 v48; // [rsp+60h] [rbp-70h]
  const char *v49; // [rsp+70h] [rbp-60h] BYREF
  __int64 v50; // [rsp+78h] [rbp-58h]
  __int64 v51; // [rsp+80h] [rbp-50h]
  __int64 v52; // [rsp+88h] [rbp-48h]
  _QWORD *v53; // [rsp+90h] [rbp-40h]
  __int64 v54; // [rsp+98h] [rbp-38h]

  v9 = a7;
  v10 = sub_2BF3F10(a1);
  v36 = sub_2BF04D0(v10);
  v11 = sub_2AAFF80((__int64)a1);
  v49 = "offset.idx";
  v12 = v11 + 96;
  v13 = v11;
  LOWORD(v53) = 259;
  if ( !v11 )
    v12 = 0;
  v14 = sub_22077B0(0xC8u);
  v16 = v14;
  if ( v14 )
  {
    v47 = v12;
    v46 = a6;
    v45 = 0;
    v48 = a7;
    sub_2AAF4A0(v14, 1, &v46, 3, &v45, v15);
    sub_9C6650(&v45);
    *(_DWORD *)(v16 + 152) = a2;
    *(_QWORD *)v16 = &unk_4A23718;
    *(_QWORD *)(v16 + 96) = &unk_4A23790;
    *(_QWORD *)(v16 + 40) = &unk_4A23758;
    *(_QWORD *)(v16 + 160) = a4;
    sub_CA0F50((__int64 *)(v16 + 168), (void **)&v49);
  }
  if ( *a9 )
  {
    v17 = (__int64 *)a9[1];
    *(_QWORD *)(v16 + 80) = *a9;
    v18 = *(_QWORD *)(v16 + 24);
    v19 = *v17;
    *(_QWORD *)(v16 + 32) = v17;
    v19 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v16 + 24) = v19 | v18 & 7;
    *(_QWORD *)(v19 + 8) = v16 + 24;
    *v17 = *v17 & 7 | (v16 + 24);
  }
  if ( !*(_DWORD *)(v13 + 56) )
    BUG();
  v20 = v16 + 96;
  v21 = *(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v13 + 48) + 40LL) + 8LL);
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = v21;
  v54 = *v21;
  if ( !v16 )
    v20 = 0;
  v41 = v20;
  v22 = sub_2BFD6A0((__int64)&v49, v20);
  if ( a5 )
  {
    v22 = *(_QWORD *)(a5 + 8);
    v23 = v41;
    v46 = *a8;
    if ( v46 )
    {
      sub_2C25AB0(&v46);
      v23 = v41;
    }
    v16 = sub_2C28A30(a9, 38, v23, v22, &v46);
    sub_9C6650(&v46);
  }
  if ( v22 != sub_2BFD6A0((__int64)&v49, a7) )
  {
    v24 = sub_2BF0570(v36);
    v25 = 0;
    if ( *(_DWORD *)(v24 + 64) == 1 )
      v25 = **(_QWORD **)(v24 + 56);
    v26 = *a9;
    v27 = a9[1];
    *a9 = v25;
    a9[1] = v25 + 112;
    v39 = v27;
    v46 = *a8;
    if ( v46 )
      sub_2C25AB0(&v46);
    v9 = sub_2C28A30(a9, 38, a7, v22, &v46);
    if ( v9 )
      v9 += 96;
    sub_9C6650(&v46);
    if ( v26 )
    {
      *a9 = v26;
      a9[1] = v39;
    }
    else
    {
      *a9 = 0;
      a9[1] = 0;
    }
  }
  if ( v16 )
    v16 += 96;
  v28 = 0;
  if ( a4 )
  {
    v28 = *(_BYTE *)(a4 + 1) >> 1;
    if ( v28 == 127 )
      v28 = -1;
  }
  v29 = sub_22077B0(0xA8u);
  v31 = v29;
  if ( v29 )
  {
    v47 = v9;
    v46 = v16;
    v44 = 0;
    v43 = v28;
    v45 = 0;
    sub_2AAF4A0(v29, 11, &v46, 2, &v45, v30);
    sub_9C6650(&v45);
    *(_BYTE *)(v31 + 152) = 5;
    *(_QWORD *)v31 = &unk_4A23258;
    *(_QWORD *)(v31 + 40) = &unk_4A23290;
    *(_QWORD *)(v31 + 96) = &unk_4A232C8;
    sub_2C1AC80((_BYTE *)(v31 + 156), &v43);
    sub_9C6650(&v44);
    *(_QWORD *)v31 = &unk_4A24130;
    *(_QWORD *)(v31 + 96) = &unk_4A241A8;
    *(_QWORD *)(v31 + 40) = &unk_4A24170;
    *(_DWORD *)(v31 + 160) = a3;
  }
  if ( *a9 )
  {
    v32 = (__int64 *)a9[1];
    *(_QWORD *)(v31 + 80) = *a9;
    v33 = *(_QWORD *)(v31 + 24);
    v34 = *v32;
    *(_QWORD *)(v31 + 32) = v32;
    v34 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v31 + 24) = v34 | v33 & 7;
    *(_QWORD *)(v34 + 8) = v31 + 24;
    *v32 = *v32 & 7 | (v31 + 24);
  }
  sub_C7D6A0(v50, 16LL * (unsigned int)v52, 8);
  return v31;
}
