// Function: sub_17CB560
// Address: 0x17cb560
//
unsigned __int64 __fastcall sub_17CB560(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _BYTE *v11; // rbx
  _QWORD *v12; // rax
  unsigned __int8 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // r13
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // r13
  __int64 *v20; // rbx
  unsigned __int64 *v21; // r14
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // rax
  __int64 v27; // r10
  _QWORD *v28; // rax
  _QWORD *v29; // r14
  unsigned __int64 *v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  unsigned __int64 result; // rax
  __int64 v38; // rax
  __int64 v39; // r10
  __int64 *v40; // r14
  __int64 v41; // rsi
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  __m128i *v46; // rsi
  __int64 v47; // rax
  _QWORD *v48; // rax
  __int64 v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // r10
  __int64 v52; // rax
  unsigned __int64 *v53; // rbx
  __int64 v54; // rax
  unsigned __int64 v55; // rcx
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  int v58; // [rsp+Ch] [rbp-104h]
  __int64 v59; // [rsp+10h] [rbp-100h]
  __int64 v60; // [rsp+18h] [rbp-F8h]
  __int64 v61; // [rsp+18h] [rbp-F8h]
  __int64 v62; // [rsp+20h] [rbp-F0h]
  __int64 v63; // [rsp+20h] [rbp-F0h]
  __int64 v64; // [rsp+20h] [rbp-F0h]
  __int64 v65; // [rsp+20h] [rbp-F0h]
  __int64 v66; // [rsp+20h] [rbp-F0h]
  __int64 v67; // [rsp+20h] [rbp-F0h]
  __int64 *v68; // [rsp+38h] [rbp-D8h] BYREF
  unsigned __int8 *v69; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v70; // [rsp+48h] [rbp-C8h]
  __int64 v71[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v72; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v73[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v74; // [rsp+80h] [rbp-90h]
  __int64 *v75; // [rsp+90h] [rbp-80h] BYREF
  __int64 v76; // [rsp+98h] [rbp-78h]
  unsigned __int64 *v77; // [rsp+A0h] [rbp-70h]
  _QWORD *v78; // [rsp+A8h] [rbp-68h]
  __int64 v79; // [rsp+B0h] [rbp-60h]
  int v80; // [rsp+B8h] [rbp-58h]
  __int64 v81; // [rsp+C0h] [rbp-50h]
  __int64 v82; // [rsp+C8h] [rbp-48h]

  v11 = (_BYTE *)sub_17CA660(a1, a2);
  v12 = (_QWORD *)sub_16498A0(a2);
  v13 = *(unsigned __int8 **)(a2 + 48);
  v75 = 0;
  v78 = v12;
  v14 = *(_QWORD *)(a2 + 40);
  v79 = 0;
  v76 = v14;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v77 = (unsigned __int64 *)(a2 + 24);
  v73[0] = v13;
  if ( v13 )
  {
    sub_1623A60((__int64)v73, (__int64)v13, 2);
    if ( v75 )
      sub_161E7C0((__int64)&v75, (__int64)v75);
    v75 = (__int64 *)v73[0];
    if ( v73[0] )
      sub_1623210((__int64)v73, v73[0], (__int64)&v75);
  }
  v15 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v72 = 257;
  v17 = sub_1643360(v78);
  v69 = (unsigned __int8 *)sub_159C470(v17, 0, 0);
  v18 = sub_1643360(v78);
  v70 = sub_159C470(v18, (__int64)v16, 0);
  if ( v11[16] > 0x10u )
  {
    v74 = 257;
    v47 = *(_QWORD *)v11;
    if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
      v47 = **(_QWORD **)(v47 + 16);
    v67 = *(_QWORD *)(v47 + 24);
    v48 = sub_1648A60(72, 3u);
    v19 = v48;
    if ( v48 )
    {
      v61 = (__int64)v48;
      v59 = (__int64)(v48 - 9);
      v49 = *(_QWORD *)v11;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16 )
        v49 = **(_QWORD **)(v49 + 16);
      v58 = *(_DWORD *)(v49 + 8) >> 8;
      v50 = (__int64 *)sub_15F9F50(v67, (__int64)&v69, 2);
      v51 = (__int64 *)sub_1646BA0(v50, v58);
      v52 = *(_QWORD *)v11;
      if ( *(_BYTE *)(*(_QWORD *)v11 + 8LL) == 16
        || (v52 = *(_QWORD *)v69, *(_BYTE *)(*(_QWORD *)v69 + 8LL) == 16)
        || (v52 = *(_QWORD *)v70, *(_BYTE *)(*(_QWORD *)v70 + 8LL) == 16) )
      {
        v51 = sub_16463B0(v51, *(_QWORD *)(v52 + 32));
      }
      sub_15F1EA0((__int64)v19, (__int64)v51, 32, v59, 3, 0);
      v19[7] = v67;
      v19[8] = sub_15F9F50(v67, (__int64)&v69, 2);
      sub_15F9CE0((__int64)v19, (__int64)v11, (__int64 *)&v69, 2, (__int64)v73);
    }
    else
    {
      v61 = 0;
    }
    sub_15FA2E0((__int64)v19, 1);
    if ( v76 )
    {
      v53 = v77;
      sub_157E9D0(v76 + 40, (__int64)v19);
      v54 = v19[3];
      v55 = *v53;
      v19[4] = v53;
      v55 &= 0xFFFFFFFFFFFFFFF8LL;
      v19[3] = v55 | v54 & 7;
      *(_QWORD *)(v55 + 8) = v19 + 3;
      *v53 = *v53 & 7 | (unsigned __int64)(v19 + 3);
    }
    sub_164B780(v61, v71);
    if ( v75 )
    {
      v68 = v75;
      sub_1623A60((__int64)&v68, (__int64)v75, 2);
      v56 = v19[6];
      if ( v56 )
        sub_161E7C0((__int64)(v19 + 6), v56);
      v57 = (unsigned __int8 *)v68;
      v19[6] = v68;
      if ( v57 )
        sub_1623210((__int64)&v68, v57, (__int64)(v19 + 6));
    }
  }
  else
  {
    BYTE4(v73[0]) = 0;
    v19 = (_QWORD *)sub_15A2E80(0, (__int64)v11, (__int64 **)&v69, 2u, 1u, (__int64)v73, 0);
  }
  v73[0] = "pgocount";
  v74 = 259;
  v20 = sub_1648A60(64, 1u);
  if ( v20 )
    sub_15F9210((__int64)v20, *(_QWORD *)(*v19 + 24LL), (__int64)v19, 0, 0, 0);
  if ( v76 )
  {
    v21 = v77;
    sub_157E9D0(v76 + 40, (__int64)v20);
    v22 = v20[3];
    v23 = *v21;
    v20[4] = (__int64)v21;
    v23 &= 0xFFFFFFFFFFFFFFF8LL;
    v20[3] = v23 | v22 & 7;
    *(_QWORD *)(v23 + 8) = v20 + 3;
    *v21 = *v21 & 7 | (unsigned __int64)(v20 + 3);
  }
  sub_164B780((__int64)v20, (__int64 *)v73);
  if ( v75 )
  {
    v71[0] = (__int64)v75;
    sub_1623A60((__int64)v71, (__int64)v75, 2);
    v24 = v20[6];
    if ( v24 )
      sub_161E7C0((__int64)(v20 + 6), v24);
    v25 = (unsigned __int8 *)v71[0];
    v20[6] = v71[0];
    if ( v25 )
      sub_1623210((__int64)v71, v25, (__int64)(v20 + 6));
  }
  v72 = 257;
  v26 = sub_1601EA0(a2);
  if ( *((_BYTE *)v20 + 16) > 0x10u || *(_BYTE *)(v26 + 16) > 0x10u )
  {
    v74 = 257;
    v38 = sub_15FB440(11, v20, v26, (__int64)v73, 0);
    v39 = v38;
    if ( v76 )
    {
      v40 = (__int64 *)v77;
      v63 = v38;
      sub_157E9D0(v76 + 40, v38);
      v39 = v63;
      v41 = *v40;
      v42 = *(_QWORD *)(v63 + 24);
      *(_QWORD *)(v63 + 32) = v40;
      v41 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v63 + 24) = v41 | v42 & 7;
      *(_QWORD *)(v41 + 8) = v63 + 24;
      *v40 = *v40 & 7 | (v63 + 24);
    }
    v64 = v39;
    sub_164B780(v39, v71);
    v27 = v64;
    if ( v75 )
    {
      v69 = (unsigned __int8 *)v75;
      sub_1623A60((__int64)&v69, (__int64)v75, 2);
      v27 = v64;
      v43 = *(_QWORD *)(v64 + 48);
      v44 = v64 + 48;
      if ( v43 )
      {
        v60 = v64;
        v65 = v64 + 48;
        sub_161E7C0(v65, v43);
        v27 = v60;
        v44 = v65;
      }
      v45 = v69;
      *(_QWORD *)(v27 + 48) = v69;
      if ( v45 )
      {
        v66 = v27;
        sub_1623210((__int64)&v69, v45, v44);
        v27 = v66;
      }
    }
  }
  else
  {
    v27 = sub_15A2B30(v20, v26, 0, 0, *(double *)a3.m128_u64, a4, a5);
  }
  v62 = v27;
  v74 = 257;
  v28 = sub_1648A60(64, 2u);
  v29 = v28;
  if ( v28 )
    sub_15F9650((__int64)v28, v62, (__int64)v19, 0, 0);
  if ( v76 )
  {
    v30 = v77;
    sub_157E9D0(v76 + 40, (__int64)v29);
    v31 = v29[3];
    v32 = *v30;
    v29[4] = v30;
    v32 &= 0xFFFFFFFFFFFFFFF8LL;
    v29[3] = v32 | v31 & 7;
    *(_QWORD *)(v32 + 8) = v29 + 3;
    *v30 = *v30 & 7 | (unsigned __int64)(v29 + 3);
  }
  sub_164B780((__int64)v29, (__int64 *)v73);
  if ( v75 )
  {
    v71[0] = (__int64)v75;
    sub_1623A60((__int64)v71, (__int64)v75, 2);
    v35 = v29[6];
    if ( v35 )
      sub_161E7C0((__int64)(v29 + 6), v35);
    v36 = (unsigned __int8 *)v71[0];
    v29[6] = v71[0];
    if ( v36 )
      sub_1623210((__int64)v71, v36, (__int64)(v29 + 6));
  }
  v71[0] = (__int64)v29;
  sub_164D160(a2, (__int64)v29, a3, a4, a5, a6, v33, v34, a9, a10);
  if ( (unsigned __int8)sub_17C5A30(a1) )
  {
    v73[0] = (unsigned __int8 *)v20;
    v46 = *(__m128i **)(a1 + 216);
    if ( v46 == *(__m128i **)(a1 + 224) )
    {
      sub_17C6FF0((const __m128i **)(a1 + 208), v46, v73, v71);
    }
    else
    {
      if ( v46 )
      {
        v46->m128i_i64[0] = (__int64)v20;
        v46->m128i_i64[1] = v71[0];
        v46 = *(__m128i **)(a1 + 216);
      }
      *(_QWORD *)(a1 + 216) = v46 + 1;
    }
  }
  result = sub_15F20C0((_QWORD *)a2);
  if ( v75 )
    return sub_161E7C0((__int64)&v75, (__int64)v75);
  return result;
}
