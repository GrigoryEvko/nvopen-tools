// Function: sub_37FDAF0
// Address: 0x37fdaf0
//
unsigned __int8 *__fastcall sub_37FDAF0(__int64 *a1, __int64 a2)
{
  const __m128i *v3; // rax
  __int128 v4; // xmm0
  unsigned __int8 *v5; // rax
  __int64 v6; // rsi
  unsigned __int8 *v7; // r14
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r15
  __int64 v10; // rbx
  unsigned __int16 v11; // ax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  unsigned int v15; // eax
  int v16; // eax
  __int64 v17; // rdx
  unsigned int v18; // ecx
  __int64 v19; // rbx
  __int64 v20; // rax
  __int16 v21; // dx
  __int64 v22; // rdx
  __int64 v23; // rdx
  char v24; // al
  int v25; // r13d
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  int v29; // r9d
  unsigned __int8 *v30; // rax
  unsigned int v31; // edx
  __int64 v32; // r9
  unsigned __int8 *v33; // r12
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int128 v39; // rax
  __int64 v40; // r9
  unsigned int v41; // edx
  int v42; // r9d
  unsigned int v43; // edx
  __int64 v44; // rax
  unsigned __int8 *v45; // rax
  __int64 v46; // r13
  unsigned int v47; // edx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdx
  __int128 v51; // rax
  __int64 v52; // r9
  unsigned int v53; // edx
  __int128 v54; // [rsp-20h] [rbp-110h]
  __int128 v55; // [rsp-20h] [rbp-110h]
  __int128 v56; // [rsp+0h] [rbp-F0h]
  int v57; // [rsp+24h] [rbp-CCh]
  unsigned int v58; // [rsp+28h] [rbp-C8h]
  int v59; // [rsp+28h] [rbp-C8h]
  __int64 v60; // [rsp+28h] [rbp-C8h]
  unsigned __int32 v61; // [rsp+30h] [rbp-C0h]
  __int64 v62; // [rsp+30h] [rbp-C0h]
  __int64 v63; // [rsp+38h] [rbp-B8h]
  __int64 v64; // [rsp+38h] [rbp-B8h]
  __int64 v65; // [rsp+38h] [rbp-B8h]
  unsigned int v66; // [rsp+38h] [rbp-B8h]
  unsigned int v67; // [rsp+40h] [rbp-B0h]
  __int16 v68; // [rsp+42h] [rbp-AEh]
  unsigned __int8 *v69; // [rsp+48h] [rbp-A8h]
  unsigned __int8 *v70; // [rsp+48h] [rbp-A8h]
  __int64 v71; // [rsp+50h] [rbp-A0h] BYREF
  int v72; // [rsp+58h] [rbp-98h]
  unsigned int v73; // [rsp+60h] [rbp-90h] BYREF
  __int64 v74; // [rsp+68h] [rbp-88h]
  unsigned int v75; // [rsp+70h] [rbp-80h] BYREF
  __int64 v76; // [rsp+78h] [rbp-78h]
  __int64 v77; // [rsp+80h] [rbp-70h] BYREF
  char v78; // [rsp+88h] [rbp-68h]
  __int64 v79; // [rsp+90h] [rbp-60h] BYREF
  __int64 v80; // [rsp+98h] [rbp-58h]
  __int64 v81; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-48h]
  __int64 v83; // [rsp+B0h] [rbp-40h]
  __int64 v84; // [rsp+B8h] [rbp-38h]

  v3 = *(const __m128i **)(a2 + 40);
  v4 = (__int128)_mm_loadu_si128(v3);
  v63 = v3->m128i_i64[0];
  v61 = v3->m128i_u32[2];
  v5 = sub_375A6A0((__int64)a1, v3[2].m128i_i64[1], v3[3].m128i_i64[0], (__m128i)v4);
  v6 = *(_QWORD *)(a2 + 80);
  v69 = v5;
  v7 = v5;
  v9 = v8;
  v58 = v8;
  v71 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v71, v6, 1);
  v72 = *(_DWORD *)(a2 + 72);
  v10 = *(_QWORD *)(v63 + 48) + 16LL * v61;
  v11 = *(_WORD *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  LOWORD(v73) = v11;
  v74 = v12;
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_43;
    v14 = 16LL * (v11 - 1);
    v13 = *(_QWORD *)&byte_444C4A0[v14];
    LOBYTE(v14) = byte_444C4A0[v14 + 8];
  }
  else
  {
    v13 = sub_3007260((__int64)&v73);
    v83 = v13;
    v84 = v14;
  }
  v81 = v13;
  LOBYTE(v82) = v14;
  v15 = sub_CA1930(&v81);
  switch ( v15 )
  {
    case 1u:
      LOWORD(v16) = 2;
      break;
    case 2u:
      LOWORD(v16) = 3;
      break;
    case 4u:
      LOWORD(v16) = 4;
      break;
    case 8u:
      LOWORD(v16) = 5;
      break;
    case 0x10u:
      LOWORD(v16) = 6;
      break;
    case 0x20u:
      LOWORD(v16) = 7;
      break;
    case 0x40u:
      LOWORD(v16) = 8;
      break;
    case 0x80u:
      LOWORD(v16) = 9;
      break;
    default:
      v16 = sub_3007020(*(_QWORD **)(a1[1] + 64), v15);
      v68 = HIWORD(v16);
      v62 = v17;
      goto LABEL_16;
  }
  v62 = 0;
LABEL_16:
  HIWORD(v18) = v68;
  LOWORD(v18) = v16;
  v67 = v18;
  v64 = v58;
  v19 = 16LL * v58;
  v20 = v19 + *((_QWORD *)v69 + 6);
  v21 = *(_WORD *)v20;
  v76 = *(_QWORD *)(v20 + 8);
  LOWORD(v75) = v21;
  if ( (_WORD)v73 )
  {
    if ( (_WORD)v73 == 1 || (unsigned __int16)(v73 - 504) <= 7u )
      goto LABEL_43;
    v44 = 16LL * ((unsigned __int16)v73 - 1);
    v23 = *(_QWORD *)&byte_444C4A0[v44];
    v24 = byte_444C4A0[v44 + 8];
  }
  else
  {
    v81 = sub_3007260((__int64)&v73);
    v82 = v22;
    v23 = v81;
    v24 = v82;
  }
  v79 = v23;
  LOBYTE(v80) = v24;
  v25 = sub_CA1930(&v79);
  if ( !(_WORD)v75 )
  {
    v26 = sub_3007260((__int64)&v75);
    v79 = v26;
    v80 = v27;
    goto LABEL_20;
  }
  if ( (_WORD)v75 == 1 || (unsigned __int16)(v75 - 504) <= 7u )
LABEL_43:
    BUG();
  v27 = 16LL * ((unsigned __int16)v75 - 1);
  v26 = *(_QWORD *)&byte_444C4A0[v27];
  LOBYTE(v27) = byte_444C4A0[v27 + 8];
LABEL_20:
  v77 = v26;
  v78 = v27;
  v28 = sub_CA1930(&v77);
  v29 = v28 - v25;
  if ( v28 - v25 > 0 )
  {
    v35 = a1[1];
    v59 = v29;
    v65 = *a1;
    v36 = sub_2E79000(*(__int64 **)(v35 + 40));
    v37 = sub_2FE6750(
            v65,
            *(unsigned __int16 *)(v19 + *((_QWORD *)v69 + 6)),
            *(_QWORD *)(v19 + *((_QWORD *)v69 + 6) + 8),
            v36);
    *(_QWORD *)&v39 = sub_3400BD0(v35, v59, (__int64)&v71, v37, v38, 0, (__m128i)v4, 0);
    *((_QWORD *)&v54 + 1) = v9;
    *(_QWORD *)&v54 = v7;
    sub_3406EB0((_QWORD *)v35, 0xC0u, (__int64)&v71, v75, v76, v40, v54, v39);
    v9 = v41 | v9 & 0xFFFFFFFF00000000LL;
    v69 = sub_33FAF80(a1[1], 216, (__int64)&v71, v67, v62, v42, (__m128i)v4);
    v64 = v43;
  }
  else if ( v28 != v25 )
  {
    v57 = v28 - v25;
    v45 = sub_33FAF80(a1[1], 215, (__int64)&v71, v73, v74, v29, (__m128i)v4);
    v46 = a1[1];
    v66 = v47;
    v60 = *a1;
    v70 = v45;
    v48 = sub_2E79000(*(__int64 **)(v46 + 40));
    v49 = sub_2FE6750(
            v60,
            *(unsigned __int16 *)(*((_QWORD *)v70 + 6) + 16LL * v66),
            *(_QWORD *)(*((_QWORD *)v70 + 6) + 16LL * v66 + 8),
            v48);
    *(_QWORD *)&v51 = sub_3400BD0(v46, -v57, (__int64)&v71, v49, v50, 0, (__m128i)v4, 0);
    v9 = v66 | v9 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v55 + 1) = v9;
    *(_QWORD *)&v55 = v70;
    v69 = sub_3406EB0((_QWORD *)v46, 0xBEu, (__int64)&v71, v67, v62, v52, v55, v51);
    v64 = v53;
  }
  v30 = sub_33FB890(a1[1], v73, v74, (__int64)v69, v64, (__m128i)v4);
  *((_QWORD *)&v56 + 1) = v31 | v64 & 0xFFFFFFFF00000000LL | v9 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v56 = v30;
  v33 = sub_3406EB0((_QWORD *)a1[1], 0x98u, (__int64)&v71, v73, v74, v32, v4, v56);
  if ( v71 )
    sub_B91220((__int64)&v71, v71);
  return v33;
}
