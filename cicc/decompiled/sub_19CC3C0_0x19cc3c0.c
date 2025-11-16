// Function: sub_19CC3C0
// Address: 0x19cc3c0
//
__int64 __fastcall sub_19CC3C0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r14
  __int64 *v15; // r12
  unsigned __int64 v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 *v23; // rdx
  unsigned __int16 v24; // si
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  __int64 v28; // r9
  __int64 *v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rax
  double v32; // xmm4_8
  double v33; // xmm5_8
  __int64 v34; // r9
  __int64 v35; // rsi
  unsigned __int8 *v36; // rsi
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned __int8 **v43; // r8
  __int64 v44; // rdx
  unsigned __int8 *v45; // rsi
  __int64 v46; // rcx
  __int64 *v47; // rdx
  unsigned __int16 v48; // si
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 *v51; // rsi
  int v52; // edi
  __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rsi
  unsigned __int64 *v56; // [rsp+8h] [rbp-F8h]
  _QWORD *v57; // [rsp+8h] [rbp-F8h]
  __int64 v58; // [rsp+8h] [rbp-F8h]
  __int64 v59; // [rsp+8h] [rbp-F8h]
  __int64 *v60; // [rsp+8h] [rbp-F8h]
  __int64 *v61; // [rsp+8h] [rbp-F8h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-E8h] BYREF
  _QWORD v63[2]; // [rsp+20h] [rbp-E0h] BYREF
  __int16 v64; // [rsp+30h] [rbp-D0h]
  __int64 v65[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v66; // [rsp+50h] [rbp-B0h]
  unsigned __int8 *v67[2]; // [rsp+60h] [rbp-A0h] BYREF
  __int16 v68; // [rsp+70h] [rbp-90h]
  unsigned __int8 *v69; // [rsp+80h] [rbp-80h] BYREF
  __int64 v70; // [rsp+88h] [rbp-78h]
  unsigned __int64 *v71; // [rsp+90h] [rbp-70h]
  __int64 v72; // [rsp+98h] [rbp-68h]
  __int64 v73; // [rsp+A0h] [rbp-60h]
  int v74; // [rsp+A8h] [rbp-58h]
  __int64 v75; // [rsp+B0h] [rbp-50h]
  __int64 v76; // [rsp+B8h] [rbp-48h]

  v10 = sub_16498A0(a1);
  v11 = *(unsigned __int8 **)(a1 + 48);
  v69 = 0;
  v72 = v10;
  v12 = *(_QWORD *)(a1 + 40);
  v73 = 0;
  v70 = v12;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v71 = (unsigned __int64 *)(a1 + 24);
  v67[0] = v11;
  if ( v11 )
  {
    sub_1623A60((__int64)v67, (__int64)v11, 2);
    if ( v69 )
      sub_161E7C0((__int64)&v69, (__int64)v69);
    v69 = v67[0];
    if ( v67[0] )
      sub_1623210((__int64)v67, v67[0], (__int64)&v69);
  }
  v13 = *(_QWORD *)(a1 - 48);
  v14 = *(_QWORD *)(a1 - 24);
  v68 = 257;
  v15 = sub_1648A60(64, 1u);
  if ( v15 )
    sub_15F9210((__int64)v15, *(_QWORD *)(*(_QWORD *)v13 + 24LL), v13, 0, 0, 0);
  if ( v70 )
  {
    v56 = v71;
    sub_157E9D0(v70 + 40, (__int64)v15);
    v16 = *v56;
    v17 = v15[3] & 7;
    v15[4] = (__int64)v56;
    v16 &= 0xFFFFFFFFFFFFFFF8LL;
    v15[3] = v16 | v17;
    *(_QWORD *)(v16 + 8) = v15 + 3;
    *v56 = *v56 & 7 | (unsigned __int64)(v15 + 3);
  }
  sub_164B780((__int64)v15, (__int64 *)v67);
  v20 = v69;
  if ( v69 )
  {
    v65[0] = (__int64)v69;
    sub_1623A60((__int64)v65, (__int64)v69, 2);
    v21 = v15[6];
    v18 = (__int64)(v15 + 6);
    if ( v21 )
    {
      sub_161E7C0((__int64)(v15 + 6), v21);
      v18 = (__int64)(v15 + 6);
    }
    v20 = (unsigned __int8 *)v65[0];
    v15[6] = v65[0];
    if ( v20 )
      sub_1623210((__int64)v65, v20, v18);
  }
  switch ( (*(unsigned __int16 *)(a1 + 18) >> 5) & 0x3FF )
  {
    case 0:
      break;
    case 1:
      v66 = 257;
      if ( *((_BYTE *)v15 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
      {
        v50 = v14;
        v68 = 257;
        v51 = v15;
        v52 = 11;
        goto LABEL_61;
      }
      v14 = sub_15A2B30(v15, v14, 0, 0, *(double *)a2.m128_u64, a3, a4);
      break;
    case 2:
      v66 = 257;
      if ( *((_BYTE *)v15 + 16) > 0x10u || *(_BYTE *)(v14 + 16) > 0x10u )
      {
        v50 = v14;
        v51 = v15;
        v68 = 257;
        v52 = 13;
        goto LABEL_61;
      }
      v14 = sub_15A2B60(v15, v14, 0, 0, *(double *)a2.m128_u64, a3, a4);
      break;
    case 3:
      v68 = 257;
      v14 = sub_1281C00((__int64 *)&v69, (__int64)v15, v14, (__int64)v67);
      break;
    case 4:
      v66 = 257;
      v64 = 257;
      v38 = sub_1281C00((__int64 *)&v69, (__int64)v15, v14, (__int64)v63);
      if ( *(_BYTE *)(v38 + 16) > 0x10u )
      {
        v68 = 257;
        v14 = sub_15FB630((__int64 *)v38, (__int64)v67, 0);
        if ( v70 )
        {
          v61 = (__int64 *)v71;
          sub_157E9D0(v70 + 40, v14);
          v53 = *v61;
          v54 = *(_QWORD *)(v14 + 24) & 7LL;
          *(_QWORD *)(v14 + 32) = v61;
          v53 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v14 + 24) = v53 | v54;
          *(_QWORD *)(v53 + 8) = v14 + 24;
          *v61 = *v61 & 7 | (v14 + 24);
        }
        sub_164B780(v14, v65);
        if ( v69 )
        {
          v62 = v69;
          sub_1623A60((__int64)&v62, (__int64)v69, 2);
          v55 = *(_QWORD *)(v14 + 48);
          v43 = &v62;
          v44 = v14 + 48;
          if ( v55 )
          {
            sub_161E7C0(v14 + 48, v55);
            v43 = &v62;
            v44 = v14 + 48;
          }
          v45 = v62;
          *(_QWORD *)(v14 + 48) = v62;
          if ( v45 )
            goto LABEL_54;
        }
      }
      else
      {
        v14 = sub_15A2B00((__int64 *)v38, *(double *)a2.m128_u64, a3, a4);
      }
      break;
    case 5:
      v66 = 257;
      if ( *(_BYTE *)(v14 + 16) > 0x10u )
        goto LABEL_64;
      if ( sub_1593BB0(v14, (__int64)v20, v18, v19) )
      {
        v14 = (__int64)v15;
      }
      else
      {
        if ( *((_BYTE *)v15 + 16) > 0x10u )
        {
LABEL_64:
          v50 = v14;
          v68 = 257;
          v51 = v15;
          v52 = 27;
LABEL_61:
          v14 = sub_15FB440(v52, v51, v50, (__int64)v67, 0);
          if ( v70 )
          {
            v39 = v70 + 40;
            v60 = (__int64 *)v71;
            goto LABEL_49;
          }
          goto LABEL_50;
        }
        v14 = sub_15A2D10(v15, v14, *(double *)a2.m128_u64, a3, a4);
      }
      break;
    case 6:
      v66 = 257;
      if ( *((_BYTE *)v15 + 16) <= 0x10u && *(_BYTE *)(v14 + 16) <= 0x10u )
      {
        v26 = sub_15A2A30((__int64 *)0x1C, v15, v14, 0, 0, *(double *)a2.m128_u64, a3, a4);
        if ( v26 )
          goto LABEL_18;
      }
      v68 = 257;
      v14 = sub_15FB440(28, v15, v14, (__int64)v67, 0);
      if ( v70 )
      {
        v39 = v70 + 40;
        v60 = (__int64 *)v71;
LABEL_49:
        sub_157E9D0(v39, v14);
        v40 = *v60;
        v41 = *(_QWORD *)(v14 + 24) & 7LL;
        *(_QWORD *)(v14 + 32) = v60;
        v40 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v14 + 24) = v40 | v41;
        *(_QWORD *)(v40 + 8) = v14 + 24;
        *v60 = *v60 & 7 | (v14 + 24);
      }
LABEL_50:
      sub_164B780(v14, v65);
      if ( v69 )
      {
        v63[0] = v69;
        sub_1623A60((__int64)v63, (__int64)v69, 2);
        v42 = *(_QWORD *)(v14 + 48);
        v43 = (unsigned __int8 **)v63;
        v44 = v14 + 48;
        if ( v42 )
        {
          sub_161E7C0(v14 + 48, v42);
          v43 = (unsigned __int8 **)v63;
          v44 = v14 + 48;
        }
        v45 = (unsigned __int8 *)v63[0];
        *(_QWORD *)(v14 + 48) = v63[0];
        if ( v45 )
LABEL_54:
          sub_1623210((__int64)v43, v45, v44);
      }
      break;
    case 7:
      v46 = v14;
      v47 = v15;
      v68 = 257;
      v48 = 40;
      v66 = 257;
      goto LABEL_56;
    case 8:
      v22 = v14;
      v23 = v15;
      v68 = 257;
      v24 = 40;
      v66 = 257;
      goto LABEL_17;
    case 9:
      v46 = v14;
      v47 = v15;
      v48 = 36;
      v68 = 257;
      v66 = 257;
LABEL_56:
      v49 = sub_12AA0C0((__int64 *)&v69, v48, v47, v46, (__int64)v65);
      v14 = sub_156B790((__int64 *)&v69, v49, v14, (__int64)v15, (__int64)v67, 0);
      break;
    case 0xA:
      v23 = v15;
      v68 = 257;
      v22 = v14;
      v66 = 257;
      v24 = 36;
LABEL_17:
      v25 = sub_12AA0C0((__int64 *)&v69, v24, v23, v22, (__int64)v65);
      v26 = sub_156B790((__int64 *)&v69, v25, (__int64)v15, v14, (__int64)v67, 0);
LABEL_18:
      v14 = v26;
      break;
  }
  v68 = 257;
  v27 = sub_1648A60(64, 2u);
  v28 = (__int64)v27;
  if ( v27 )
  {
    v57 = v27;
    sub_15F9650((__int64)v27, v14, v13, 0, 0);
    v28 = (__int64)v57;
  }
  if ( v70 )
  {
    v29 = (__int64 *)v71;
    v58 = v28;
    sub_157E9D0(v70 + 40, v28);
    v28 = v58;
    v30 = *v29;
    v31 = *(_QWORD *)(v58 + 24);
    *(_QWORD *)(v58 + 32) = v29;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v58 + 24) = v30 | v31 & 7;
    *(_QWORD *)(v30 + 8) = v58 + 24;
    *v29 = *v29 & 7 | (v58 + 24);
  }
  v59 = v28;
  sub_164B780(v28, (__int64 *)v67);
  if ( v69 )
  {
    v65[0] = (__int64)v69;
    sub_1623A60((__int64)v65, (__int64)v69, 2);
    v34 = v59;
    v35 = *(_QWORD *)(v59 + 48);
    if ( v35 )
    {
      sub_161E7C0(v59 + 48, v35);
      v34 = v59;
    }
    v36 = (unsigned __int8 *)v65[0];
    *(_QWORD *)(v34 + 48) = v65[0];
    if ( v36 )
      sub_1623210((__int64)v65, v36, v59 + 48);
  }
  sub_164D160(a1, (__int64)v15, a2, a3, a4, a5, v32, v33, a8, a9);
  sub_15F20C0((_QWORD *)a1);
  if ( v69 )
    sub_161E7C0((__int64)&v69, (__int64)v69);
  return 1;
}
