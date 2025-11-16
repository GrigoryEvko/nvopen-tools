// Function: sub_33ED290
// Address: 0x33ed290
//
_QWORD *__fastcall sub_33ED290(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        unsigned __int64 a6,
        char a7,
        int a8)
{
  __int64 v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  int v14; // r10d
  unsigned int v15; // r14d
  int v16; // r14d
  __m128i *v17; // rax
  int v18; // edx
  __int64 v19; // r8
  __int64 v20; // rax
  int v21; // r10d
  unsigned __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rax
  _QWORD *v29; // r12
  unsigned __int64 v31; // r8
  int v32; // r9d
  __int64 v33; // rsi
  unsigned __int8 *v34; // rsi
  __int64 v35; // rcx
  unsigned __int64 v36; // rax
  __int64 v37; // rax
  int v38; // [rsp+10h] [rbp-100h]
  __m128i *v39; // [rsp+18h] [rbp-F8h]
  int v40; // [rsp+24h] [rbp-ECh]
  int v41; // [rsp+24h] [rbp-ECh]
  int v42; // [rsp+24h] [rbp-ECh]
  unsigned __int64 v44; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v45; // [rsp+30h] [rbp-E0h]
  int v46; // [rsp+30h] [rbp-E0h]
  int v48; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v49; // [rsp+38h] [rbp-D8h]
  __int64 v50; // [rsp+38h] [rbp-D8h]
  __int64 *v51; // [rsp+40h] [rbp-D0h] BYREF
  unsigned __int8 *v52; // [rsp+48h] [rbp-C8h] BYREF
  _BYTE *v53; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v54; // [rsp+58h] [rbp-B8h]
  _BYTE v55[176]; // [rsp+60h] [rbp-B0h] BYREF

  v11 = sub_2E79000(*(__int64 **)(a1 + 40));
  v12 = sub_AE43A0(v11, *(_QWORD *)(a2 + 8));
  v13 = a5;
  if ( v12 > 0x3F )
    goto LABEL_4;
  v14 = v12;
  if ( v12 )
  {
    a6 = (__int64)(a6 << (64 - (unsigned __int8)v12)) >> (64 - (unsigned __int8)v12);
LABEL_4:
    v14 = a6;
    v44 = HIDWORD(a6);
    goto LABEL_5;
  }
  LODWORD(v44) = 0;
  a6 = 0;
LABEL_5:
  v15 = a7 == 0 ? 0xFFFFFFE8 : 0;
  if ( (*(_BYTE *)(a2 + 33) & 0x1C) != 0 )
    v16 = v15 + 38;
  else
    v16 = v15 + 37;
  v40 = v14;
  v17 = sub_33ED250(a1, a4, v13);
  v38 = v18;
  v54 = 0x2000000000LL;
  v39 = v17;
  v53 = v55;
  sub_33C9670((__int64)&v53, v16, (unsigned __int64)v17, 0, 0, (__int64)&v53);
  v20 = (unsigned int)v54;
  v21 = v40;
  v22 = (unsigned int)v54 + 1LL;
  if ( v22 > HIDWORD(v54) )
  {
    sub_C8D5F0((__int64)&v53, v55, v22, 4u, v19, (__int64)&v53);
    v20 = (unsigned int)v54;
    v21 = v40;
  }
  v23 = HIDWORD(a2);
  *(_DWORD *)&v53[4 * v20] = a2;
  LODWORD(v54) = v54 + 1;
  v24 = (unsigned int)v54;
  if ( (unsigned __int64)(unsigned int)v54 + 1 > HIDWORD(v54) )
  {
    v41 = v21;
    sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 4u, v23, (__int64)&v53);
    v24 = (unsigned int)v54;
    v23 = HIDWORD(a2);
    v21 = v41;
  }
  *(_DWORD *)&v53[4 * v24] = v23;
  LODWORD(v54) = v54 + 1;
  v25 = (unsigned int)v54;
  if ( (unsigned __int64)(unsigned int)v54 + 1 > HIDWORD(v54) )
  {
    v42 = v21;
    sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 4u, v23, (__int64)&v53);
    v25 = (unsigned int)v54;
    v21 = v42;
  }
  *(_DWORD *)&v53[4 * v25] = v21;
  LODWORD(v54) = v54 + 1;
  v26 = (unsigned int)v54;
  if ( (unsigned __int64)(unsigned int)v54 + 1 > HIDWORD(v54) )
  {
    sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 4u, v23, (__int64)&v53);
    v26 = (unsigned int)v54;
  }
  *(_DWORD *)&v53[4 * v26] = v44;
  LODWORD(v54) = v54 + 1;
  v27 = (unsigned int)v54;
  if ( (unsigned __int64)(unsigned int)v54 + 1 > HIDWORD(v54) )
  {
    sub_C8D5F0((__int64)&v53, v55, (unsigned int)v54 + 1LL, 4u, v23, (__int64)&v53);
    v27 = (unsigned int)v54;
  }
  *(_DWORD *)&v53[4 * v27] = a8;
  LODWORD(v54) = v54 + 1;
  v51 = 0;
  v28 = sub_33CCCF0(a1, (__int64)&v53, a3, (__int64 *)&v51);
  if ( v28 )
  {
    v29 = v28;
    goto LABEL_19;
  }
  v31 = *(_QWORD *)(a1 + 416);
  v32 = *(_DWORD *)(a3 + 8);
  if ( v31 )
  {
    *(_QWORD *)(a1 + 416) = *(_QWORD *)v31;
LABEL_26:
    v33 = *(_QWORD *)a3;
    v52 = (unsigned __int8 *)v33;
    if ( v33 )
    {
      v45 = v31;
      v48 = v32;
      sub_B96E90((__int64)&v52, v33, 1);
      v31 = v45;
      v32 = v48;
    }
    *(_QWORD *)v31 = 0;
    v34 = v52;
    *(_QWORD *)(v31 + 8) = 0;
    *(_QWORD *)(v31 + 48) = v39;
    *(_QWORD *)(v31 + 16) = 0;
    *(_DWORD *)(v31 + 24) = v16;
    *(_DWORD *)(v31 + 28) = 0;
    *(_WORD *)(v31 + 34) = -1;
    *(_DWORD *)(v31 + 36) = -1;
    *(_QWORD *)(v31 + 40) = 0;
    *(_QWORD *)(v31 + 56) = 0;
    *(_DWORD *)(v31 + 64) = 0;
    *(_DWORD *)(v31 + 68) = v38;
    *(_DWORD *)(v31 + 72) = v32;
    *(_QWORD *)(v31 + 80) = v34;
    if ( v34 )
    {
      v49 = v31;
      sub_B976B0((__int64)&v52, v34, v31 + 80);
      v31 = v49;
      *(_QWORD *)(v49 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v49 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v31 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v31 + 32) = 0;
    }
    *(_QWORD *)(v31 + 96) = a2;
    *(_QWORD *)(v31 + 104) = a6;
    *(_DWORD *)(v31 + 112) = a8;
    goto LABEL_31;
  }
  v35 = *(_QWORD *)(a1 + 424);
  *(_QWORD *)(a1 + 504) += 120LL;
  v36 = (v35 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_QWORD *)(a1 + 432) < v36 + 120 || !v35 )
  {
    v46 = v32;
    v37 = sub_9D1E70(a1 + 424, 120, 120, 3);
    v32 = v46;
    v31 = v37;
    goto LABEL_26;
  }
  *(_QWORD *)(a1 + 424) = v36 + 120;
  if ( v36 )
  {
    v31 = (v35 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_26;
  }
LABEL_31:
  v50 = v31;
  sub_C657C0((__int64 *)(a1 + 520), (__int64 *)v31, v51, (__int64)off_4A367D0);
  sub_33CC420(a1, v50);
  v29 = (_QWORD *)v50;
LABEL_19:
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return v29;
}
