// Function: sub_19D3100
// Address: 0x19d3100
//
__int64 __fastcall sub_19D3100(__int64 *a1, __int64 a2, __int64 a3, double a4, double a5, double a6)
{
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rax
  char v11; // dl
  unsigned __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v15; // r13
  __int64 v16; // r15
  unsigned int v17; // eax
  unsigned int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // rax
  unsigned __int8 *v21; // rsi
  __int64 v22; // rax
  unsigned int v23; // edx
  unsigned int v24; // eax
  __int64 **v25; // rdx
  unsigned __int8 *v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 *v30; // r15
  __int64 v31; // rcx
  __int64 v32; // rbx
  _QWORD *v33; // rax
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 *v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 **v54; // rax
  __int64 v55; // rbx
  __int64 *v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rsi
  __int64 v59; // rax
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  __int64 **v62; // rdx
  __int64 v63; // rax
  __int64 *v64; // rbx
  __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rbx
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // rax
  __int64 *v71; // rbx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rsi
  __int64 *v75; // [rsp+0h] [rbp-110h]
  __int64 *v76; // [rsp+8h] [rbp-108h]
  __int64 *v77; // [rsp+10h] [rbp-100h]
  __int64 v78; // [rsp+10h] [rbp-100h]
  __int64 *v79; // [rsp+10h] [rbp-100h]
  __int64 *v80; // [rsp+10h] [rbp-100h]
  __int64 *v81; // [rsp+10h] [rbp-100h]
  _BYTE *v82; // [rsp+18h] [rbp-F8h]
  unsigned int v83; // [rsp+24h] [rbp-ECh]
  unsigned int v84; // [rsp+30h] [rbp-E0h]
  __int64 v85; // [rsp+38h] [rbp-D8h]
  __int64 v86; // [rsp+38h] [rbp-D8h]
  __int64 v87; // [rsp+48h] [rbp-C8h] BYREF
  __int64 v88[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v89; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v90[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int16 v91; // [rsp+80h] [rbp-90h]
  __m128i v92; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v93; // [rsp+A0h] [rbp-70h]
  __int64 v94; // [rsp+A8h] [rbp-68h]
  __int64 v95; // [rsp+B0h] [rbp-60h]
  int v96; // [rsp+B8h] [rbp-58h]
  __int64 v97; // [rsp+C0h] [rbp-50h]
  __int64 v98; // [rsp+C8h] [rbp-48h]

  v8 = *a1;
  v9 = *(_QWORD *)(a2 + 40);
  v85 = a2 + 24;
  sub_141F800(&v92, a3);
  v10 = sub_141C340(v8, &v92, 0, (_QWORD *)(a2 + 24), v9, 0, 0, 0);
  v11 = v10;
  v12 = v10 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v11 & 7u) >= 3 )
    v12 = 0;
  v13 = 0;
  if ( a3 == v12 )
  {
    v82 = *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v15 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v16 = *(_QWORD *)(a3 + 24 * (2LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    v84 = sub_15603A0((_QWORD *)(a2 + 56), 0);
    v17 = sub_15603A0((_QWORD *)(a3 + 56), 0);
    v18 = v84;
    v83 = 1;
    if ( v84 < v17 )
      v18 = v17;
    if ( v18 > 1 && *(_BYTE *)(v15 + 16) == 13 )
    {
      v19 = *(_QWORD **)(v15 + 24);
      if ( *(_DWORD *)(v15 + 32) > 0x40u )
        v19 = (_QWORD *)*v19;
      v83 = ((unsigned int)v19 | v18) & -((unsigned int)v19 | v18);
    }
    v20 = sub_16498A0(a2);
    v21 = *(unsigned __int8 **)(a2 + 48);
    v92.m128i_i64[0] = 0;
    v94 = v20;
    v22 = *(_QWORD *)(a2 + 40);
    v95 = 0;
    v92.m128i_i64[1] = v22;
    v96 = 0;
    v93 = (__int64 *)v85;
    v97 = 0;
    v98 = 0;
    v90[0] = v21;
    if ( v21 )
    {
      sub_1623A60((__int64)v90, (__int64)v21, 2);
      if ( v92.m128i_i64[0] )
        sub_161E7C0((__int64)&v92, v92.m128i_i64[0]);
      v92.m128i_i64[0] = (__int64)v90[0];
      if ( v90[0] )
        sub_1623210((__int64)v90, v90[0], (__int64)&v92);
    }
    if ( *(_QWORD *)v16 == *(_QWORD *)v15 )
      goto LABEL_22;
    v23 = *(_DWORD *)(*(_QWORD *)v16 + 8LL);
    v24 = *(_DWORD *)(*(_QWORD *)v15 + 8LL);
    v89 = 257;
    if ( v23 >> 8 <= v24 >> 8 )
    {
      v62 = *(__int64 ***)v15;
      if ( *(_QWORD *)v15 == *(_QWORD *)v16 )
        goto LABEL_22;
      if ( *(_BYTE *)(v16 + 16) <= 0x10u )
      {
        v16 = sub_15A46C0(37, (__int64 ***)v16, v62, 0);
        goto LABEL_22;
      }
      v91 = 257;
      v63 = sub_15FDBD0(37, v16, (__int64)v62, (__int64)v90, 0);
      v16 = v63;
      if ( v92.m128i_i64[1] )
      {
        v64 = v93;
        sub_157E9D0(v92.m128i_i64[1] + 40, v63);
        v65 = *(_QWORD *)(v16 + 24);
        v66 = *v64;
        *(_QWORD *)(v16 + 32) = v64;
        v66 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v16 + 24) = v66 | v65 & 7;
        *(_QWORD *)(v66 + 8) = v16 + 24;
        *v64 = *v64 & 7 | (v16 + 24);
      }
      sub_164B780(v16, v88);
      if ( !v92.m128i_i64[0] )
        goto LABEL_22;
      v87 = v92.m128i_i64[0];
      v67 = v16 + 48;
      sub_1623A60((__int64)&v87, v92.m128i_i64[0], 2);
      v68 = *(_QWORD *)(v16 + 48);
      if ( v68 )
        sub_161E7C0(v16 + 48, v68);
      v69 = (unsigned __int8 *)v87;
      *(_QWORD *)(v16 + 48) = v87;
      if ( !v69 )
        goto LABEL_22;
    }
    else
    {
      v25 = *(__int64 ***)v16;
      if ( *(_QWORD *)v16 == *(_QWORD *)v15 )
        goto LABEL_22;
      if ( *(_BYTE *)(v15 + 16) <= 0x10u )
      {
        v15 = sub_15A46C0(37, (__int64 ***)v15, v25, 0);
        goto LABEL_22;
      }
      v91 = 257;
      v70 = sub_15FDBD0(37, v15, (__int64)v25, (__int64)v90, 0);
      v15 = v70;
      if ( v92.m128i_i64[1] )
      {
        v71 = v93;
        sub_157E9D0(v92.m128i_i64[1] + 40, v70);
        v72 = *(_QWORD *)(v15 + 24);
        v73 = *v71;
        *(_QWORD *)(v15 + 32) = v71;
        v73 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v15 + 24) = v73 | v72 & 7;
        *(_QWORD *)(v73 + 8) = v15 + 24;
        *v71 = *v71 & 7 | (v15 + 24);
      }
      sub_164B780(v15, v88);
      if ( !v92.m128i_i64[0] )
        goto LABEL_22;
      v87 = v92.m128i_i64[0];
      v67 = v15 + 48;
      sub_1623A60((__int64)&v87, v92.m128i_i64[0], 2);
      v74 = *(_QWORD *)(v15 + 48);
      if ( v74 )
        sub_161E7C0(v15 + 48, v74);
      v69 = (unsigned __int8 *)v87;
      *(_QWORD *)(v15 + 48) = v87;
      if ( !v69 )
        goto LABEL_22;
    }
    sub_1623210((__int64)&v87, v69, v67);
LABEL_22:
    v89 = 257;
    if ( *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
    {
      v91 = 257;
      v86 = (__int64)sub_1648A60(56, 2u);
      if ( v86 )
      {
        v54 = *(__int64 ***)v16;
        v55 = v86;
        if ( *(_BYTE *)(*(_QWORD *)v16 + 8LL) == 16 )
        {
          v80 = v54[4];
          v56 = (__int64 *)sub_1643320(*v54);
          v57 = (__int64)sub_16463B0(v56, (unsigned int)v80);
        }
        else
        {
          v57 = sub_1643320(*v54);
        }
        sub_15FEC10(v86, v57, 51, 37, v16, v15, (__int64)v90, 0);
      }
      else
      {
        v55 = 0;
      }
      if ( v92.m128i_i64[1] )
      {
        v81 = v93;
        sub_157E9D0(v92.m128i_i64[1] + 40, v86);
        v58 = *v81;
        v59 = *(_QWORD *)(v86 + 24);
        *(_QWORD *)(v86 + 32) = v81;
        v58 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v86 + 24) = v58 | v59 & 7;
        *(_QWORD *)(v58 + 8) = v86 + 24;
        *v81 = *v81 & 7 | (v86 + 24);
      }
      sub_164B780(v55, v88);
      if ( v92.m128i_i64[0] )
      {
        v87 = v92.m128i_i64[0];
        sub_1623A60((__int64)&v87, v92.m128i_i64[0], 2);
        v60 = *(_QWORD *)(v86 + 48);
        if ( v60 )
          sub_161E7C0(v86 + 48, v60);
        v61 = (unsigned __int8 *)v87;
        *(_QWORD *)(v86 + 48) = v87;
        if ( v61 )
          sub_1623210((__int64)&v87, v61, v86 + 48);
      }
    }
    else
    {
      v86 = sub_15A37B0(0x25u, (_QWORD *)v16, (_QWORD *)v15, 0);
    }
    v89 = 257;
    if ( *(_BYTE *)(v16 + 16) > 0x10u || *(_BYTE *)(v15 + 16) > 0x10u )
    {
      v91 = 257;
      v50 = sub_15FB440(13, (__int64 *)v16, v15, (__int64)v90, 0);
      v28 = v50;
      if ( v92.m128i_i64[1] )
      {
        v79 = v93;
        sub_157E9D0(v92.m128i_i64[1] + 40, v50);
        v51 = *v79;
        v52 = *(_QWORD *)(v28 + 24) & 7LL;
        *(_QWORD *)(v28 + 32) = v79;
        v51 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v28 + 24) = v51 | v52;
        *(_QWORD *)(v51 + 8) = v28 + 24;
        *v79 = *v79 & 7 | (v28 + 24);
      }
      sub_164B780(v28, v88);
      v26 = (unsigned __int8 *)v92.m128i_i64[0];
      if ( v92.m128i_i64[0] )
      {
        v87 = v92.m128i_i64[0];
        sub_1623A60((__int64)&v87, v92.m128i_i64[0], 2);
        v53 = *(_QWORD *)(v28 + 48);
        v27 = v28 + 48;
        if ( v53 )
        {
          sub_161E7C0(v28 + 48, v53);
          v27 = v28 + 48;
        }
        v26 = (unsigned __int8 *)v87;
        *(_QWORD *)(v28 + 48) = v87;
        if ( v26 )
          sub_1623210((__int64)&v87, v26, v27);
      }
    }
    else
    {
      v26 = (unsigned __int8 *)v15;
      v28 = sub_15A2B60((__int64 *)v16, v15, 0, 0, a4, a5, a6);
    }
    v89 = 257;
    v29 = sub_15A06D0(*(__int64 ***)v16, (__int64)v26, v27, 257);
    if ( *(_BYTE *)(v86 + 16) > 0x10u || *(_BYTE *)(v29 + 16) > 0x10u || *(_BYTE *)(v28 + 16) > 0x10u )
    {
      v77 = (__int64 *)v29;
      v91 = 257;
      v34 = sub_1648A60(56, 3u);
      v30 = v34;
      if ( v34 )
      {
        v35 = *v77;
        v75 = v34 - 9;
        v76 = v77;
        v78 = (__int64)v34;
        sub_15F1EA0((__int64)v34, v35, 55, (__int64)(v34 - 9), 3, 0);
        if ( *(v30 - 9) )
        {
          v36 = *(v30 - 8);
          v37 = *(v30 - 7) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v37 = v36;
          if ( v36 )
            *(_QWORD *)(v36 + 16) = *(_QWORD *)(v36 + 16) & 3LL | v37;
        }
        *(v30 - 9) = v86;
        v38 = *(_QWORD *)(v86 + 8);
        *(v30 - 8) = v38;
        if ( v38 )
          *(_QWORD *)(v38 + 16) = (unsigned __int64)(v30 - 8) | *(_QWORD *)(v38 + 16) & 3LL;
        *(v30 - 7) = (v86 + 8) | *(v30 - 7) & 3;
        *(_QWORD *)(v86 + 8) = v75;
        if ( *(v30 - 6) )
        {
          v39 = *(v30 - 5);
          v40 = *(v30 - 4) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v40 = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 16) = *(_QWORD *)(v39 + 16) & 3LL | v40;
        }
        *(v30 - 6) = (__int64)v76;
        v41 = v76[1];
        *(v30 - 5) = v41;
        if ( v41 )
          *(_QWORD *)(v41 + 16) = (unsigned __int64)(v30 - 5) | *(_QWORD *)(v41 + 16) & 3LL;
        *(v30 - 4) = (unsigned __int64)(v76 + 1) | *(v30 - 4) & 3;
        v76[1] = (__int64)(v30 - 6);
        if ( *(v30 - 3) )
        {
          v42 = *(v30 - 2);
          v43 = *(v30 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v43 = v42;
          if ( v42 )
            *(_QWORD *)(v42 + 16) = *(_QWORD *)(v42 + 16) & 3LL | v43;
        }
        *(v30 - 3) = v28;
        if ( v28 )
        {
          v44 = *(_QWORD *)(v28 + 8);
          *(v30 - 2) = v44;
          if ( v44 )
            *(_QWORD *)(v44 + 16) = (unsigned __int64)(v30 - 2) | *(_QWORD *)(v44 + 16) & 3LL;
          *(v30 - 1) = (v28 + 8) | *(v30 - 1) & 3;
          *(_QWORD *)(v28 + 8) = v30 - 3;
        }
        sub_164B780((__int64)v30, (__int64 *)v90);
      }
      else
      {
        v78 = 0;
      }
      if ( v92.m128i_i64[1] )
      {
        v45 = (unsigned __int64 *)v93;
        sub_157E9D0(v92.m128i_i64[1] + 40, (__int64)v30);
        v46 = v30[3];
        v47 = *v45;
        v30[4] = (__int64)v45;
        v47 &= 0xFFFFFFFFFFFFFFF8LL;
        v30[3] = v47 | v46 & 7;
        *(_QWORD *)(v47 + 8) = v30 + 3;
        *v45 = *v45 & 7 | (unsigned __int64)(v30 + 3);
      }
      sub_164B780(v78, v88);
      if ( v92.m128i_i64[0] )
      {
        v90[0] = (unsigned __int8 *)v92.m128i_i64[0];
        sub_1623A60((__int64)v90, v92.m128i_i64[0], 2);
        v48 = v30[6];
        if ( v48 )
          sub_161E7C0((__int64)(v30 + 6), v48);
        v49 = v90[0];
        v30[6] = (__int64)v90[0];
        if ( v49 )
          sub_1623210((__int64)v90, v49, (__int64)(v30 + 6));
      }
    }
    else
    {
      v30 = (__int64 *)sub_15A2DC0(v86, (__int64 *)v29, v28, 0);
    }
    v31 = v15;
    v13 = 1;
    v32 = *(_QWORD *)(a3 + 24 * (1LL - (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)));
    v91 = 257;
    v33 = (_QWORD *)sub_12815B0(v92.m128i_i64, 0, v82, v31, (__int64)v90);
    sub_15E7280(v92.m128i_i64, v33, v32, v30, v83, 0, 0, 0, 0);
    sub_14191F0(*a1, a3);
    sub_15F20C0((_QWORD *)a3);
    if ( v92.m128i_i64[0] )
      sub_161E7C0((__int64)&v92, v92.m128i_i64[0]);
  }
  return v13;
}
