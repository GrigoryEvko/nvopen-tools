// Function: sub_375AFE0
// Address: 0x375afe0
//
unsigned __int8 *__fastcall sub_375AFE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  unsigned int v6; // r14d
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v12; // rsi
  __int64 v13; // r12
  unsigned __int16 v14; // cx
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned __int16 v17; // ax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rbx
  __int64 v21; // rdx
  char v22; // r12
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // r9
  __int64 **v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rdx
  int v36; // r9d
  unsigned int v37; // edx
  int v38; // r9d
  unsigned int v39; // edx
  _QWORD *v40; // r15
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  __int128 v44; // rax
  __int64 v45; // r9
  unsigned int v46; // edx
  __int64 v47; // r9
  unsigned __int8 *v48; // r14
  __int16 v50; // ax
  __int128 v51; // [rsp-10h] [rbp-110h]
  unsigned int v52; // [rsp+0h] [rbp-100h]
  unsigned int v53; // [rsp+8h] [rbp-F8h]
  unsigned __int16 v54; // [rsp+8h] [rbp-F8h]
  __int64 v55; // [rsp+8h] [rbp-F8h]
  __int128 v56; // [rsp+10h] [rbp-F0h]
  __int128 v58; // [rsp+20h] [rbp-E0h]
  __int64 v60; // [rsp+50h] [rbp-B0h] BYREF
  int v61; // [rsp+58h] [rbp-A8h]
  __int64 v62; // [rsp+60h] [rbp-A0h] BYREF
  int v63; // [rsp+68h] [rbp-98h]
  unsigned __int16 v64; // [rsp+70h] [rbp-90h] BYREF
  __int64 v65; // [rsp+78h] [rbp-88h]
  unsigned __int16 v66; // [rsp+80h] [rbp-80h] BYREF
  __int64 v67; // [rsp+88h] [rbp-78h]
  __int64 v68; // [rsp+90h] [rbp-70h] BYREF
  char v69; // [rsp+98h] [rbp-68h]
  __int64 v70; // [rsp+A0h] [rbp-60h]
  __int64 v71; // [rsp+A8h] [rbp-58h]
  __int64 v72; // [rsp+B0h] [rbp-50h]
  __int64 v73; // [rsp+B8h] [rbp-48h]
  __int64 v74; // [rsp+C0h] [rbp-40h] BYREF
  __int64 v75; // [rsp+C8h] [rbp-38h]

  *(_QWORD *)&v56 = a2;
  *((_QWORD *)&v56 + 1) = a3;
  *(_QWORD *)&v58 = a4;
  *((_QWORD *)&v58 + 1) = a5;
  v9 = (unsigned int)a3;
  v10 = *(_QWORD *)(a4 + 80);
  v53 = a5;
  v60 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v60, v10, 1);
  v12 = *(_QWORD *)(a2 + 80);
  v61 = *(_DWORD *)(a4 + 72);
  v62 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v62, v12, 1);
  v13 = *(_QWORD *)(a2 + 48) + 16 * v9;
  v63 = *(_DWORD *)(a2 + 72);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  v16 = *(_QWORD *)(a4 + 48) + 16LL * v53;
  v64 = *(_WORD *)v13;
  v65 = v15;
  v17 = *(_WORD *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v66 = v17;
  v67 = v18;
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      goto LABEL_50;
    v20 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
    v22 = byte_444C4A0[16 * v17 - 8];
  }
  else
  {
    v54 = v14;
    v19 = sub_3007260((__int64)&v66);
    v14 = v54;
    v72 = v19;
    v20 = v19;
    v73 = v21;
    v22 = v21;
  }
  if ( v14 )
  {
    if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
      goto LABEL_50;
    v23 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
    LOBYTE(v24) = byte_444C4A0[16 * v14 - 8];
  }
  else
  {
    v23 = sub_3007260((__int64)&v64);
    v70 = v23;
    v71 = v24;
  }
  v25 = v20 + v23;
  if ( v20 )
    LOBYTE(v24) = v22;
  v74 = v25;
  LOBYTE(v75) = v24;
  v26 = sub_CA1930(&v74);
  v28 = (__int64 **)a1[1];
  switch ( v26 )
  {
    case 1u:
      v50 = 2;
      goto LABEL_29;
    case 2u:
      v50 = 3;
      goto LABEL_29;
    case 4u:
      v50 = 4;
      goto LABEL_29;
    case 8u:
      v50 = 5;
      goto LABEL_29;
    case 0x10u:
      v50 = 6;
      goto LABEL_29;
    case 0x20u:
      v50 = 7;
      goto LABEL_29;
    case 0x40u:
      v50 = 8;
LABEL_29:
      v31 = 0;
      LOWORD(v6) = v50;
      if ( !(_BYTE)qword_5050E88 )
        goto LABEL_20;
      goto LABEL_30;
    case 0x80u:
      v50 = 9;
      goto LABEL_29;
  }
  v29 = sub_3007020(v28[8], v26);
  v28 = (__int64 **)a1[1];
  v31 = v30;
  v6 = v29;
  if ( !(_BYTE)qword_5050E88 )
  {
LABEL_20:
    v32 = *a1;
    v33 = sub_2E79000(v28[5]);
    v34 = sub_2FE6750(v32, v6, v31, v33);
    v55 = v35;
    v52 = v34;
    *(_QWORD *)&v56 = sub_33FAF80(a1[1], 214, (__int64)&v62, v6, v31, v36, a6);
    *((_QWORD *)&v56 + 1) = v37 | *((_QWORD *)&v56 + 1) & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v58 = sub_33FAF80(a1[1], 215, (__int64)&v60, v6, v31, v38, a6);
    v40 = (_QWORD *)a1[1];
    *((_QWORD *)&v58 + 1) = v39 | *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL;
    if ( !v64 )
    {
      v41 = sub_3007260((__int64)&v64);
      v74 = v41;
      v75 = v42;
LABEL_22:
      v69 = v42;
      v68 = v41;
      v43 = sub_CA1930(&v68);
      *(_QWORD *)&v44 = sub_3400BD0((__int64)v40, v43, (__int64)&v60, v52, v55, 0, a6, 0);
      *(_QWORD *)&v58 = sub_3406EB0(v40, 0xBEu, (__int64)&v60, v6, v31, v45, v58, v44);
      *((_QWORD *)&v51 + 1) = v46 | *((_QWORD *)&v58 + 1) & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v51 = v58;
      v48 = sub_3406EB0((_QWORD *)a1[1], 0xBBu, (__int64)&v60, v6, v31, v47, v56, v51);
      goto LABEL_23;
    }
    if ( v64 != 1 && (unsigned __int16)(v64 - 504) > 7u )
    {
      v42 = 16LL * (v64 - 1);
      v41 = *(_QWORD *)&byte_444C4A0[v42];
      LOBYTE(v42) = byte_444C4A0[v42 + 8];
      goto LABEL_22;
    }
LABEL_50:
    BUG();
  }
LABEL_30:
  if ( (unsigned int)(*((_DWORD *)*v28 + 136) - 42) > 1 || v64 != v66 || (unsigned __int16)(v64 - 6) > 1u )
    goto LABEL_20;
  v48 = sub_3406EB0(v28, 0x36u, (__int64)&v60, v6, v31, v27, v56, v58);
LABEL_23:
  if ( v62 )
    sub_B91220((__int64)&v62, v62);
  if ( v60 )
    sub_B91220((__int64)&v60, v60);
  return v48;
}
