// Function: sub_19CCCA0
// Address: 0x19ccca0
//
__int64 __fastcall sub_19CCCA0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9,
        __int64 a10,
        __int64 a11)
{
  __int64 v11; // r14
  int v12; // r12d
  __int64 v13; // rbx
  char v14; // al
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rsi
  __int64 v18; // r15
  __int64 *v19; // r12
  unsigned __int64 v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rdx
  unsigned __int8 *v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // r9
  __int64 v27; // r10
  __int64 *v28; // r15
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rax
  __int64 *v35; // r15
  __int64 v36; // rbx
  _QWORD *v37; // r15
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // rsi
  __int64 v41; // rdx
  _QWORD *v43; // rax
  __int64 v44; // r12
  unsigned __int64 *v45; // rbx
  __int64 v46; // rax
  unsigned __int64 v47; // rcx
  __int64 v48; // rsi
  unsigned __int8 *v49; // rsi
  _QWORD *v50; // rax
  __int64 *v51; // r12
  __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // rsi
  unsigned __int8 *v55; // rsi
  char v56; // [rsp+17h] [rbp-119h]
  __int64 v57; // [rsp+18h] [rbp-118h]
  unsigned __int64 *v58; // [rsp+28h] [rbp-108h]
  __int64 v59; // [rsp+30h] [rbp-100h]
  _QWORD *v60; // [rsp+38h] [rbp-F8h]
  __int64 v61; // [rsp+40h] [rbp-F0h]
  __int64 v62; // [rsp+40h] [rbp-F0h]
  _QWORD *v63; // [rsp+40h] [rbp-F0h]
  __int64 v64; // [rsp+40h] [rbp-F0h]
  __int64 v65; // [rsp+40h] [rbp-F0h]
  __int64 v66; // [rsp+40h] [rbp-F0h]
  __int64 v67; // [rsp+48h] [rbp-E8h]
  __int64 v68; // [rsp+48h] [rbp-E8h]
  __int64 v69; // [rsp+58h] [rbp-D8h]
  int v70; // [rsp+64h] [rbp-CCh] BYREF
  unsigned __int8 *v71; // [rsp+68h] [rbp-C8h] BYREF
  __int64 v72[2]; // [rsp+70h] [rbp-C0h] BYREF
  __int16 v73; // [rsp+80h] [rbp-B0h]
  unsigned __int8 *v74[2]; // [rsp+90h] [rbp-A0h] BYREF
  __int16 v75; // [rsp+A0h] [rbp-90h]
  unsigned __int8 *v76; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v77; // [rsp+B8h] [rbp-78h]
  unsigned __int64 *v78; // [rsp+C0h] [rbp-70h]
  __int64 v79; // [rsp+C8h] [rbp-68h]
  __int64 v80; // [rsp+D0h] [rbp-60h]
  int v81; // [rsp+D8h] [rbp-58h]
  __int64 v82; // [rsp+E0h] [rbp-50h]
  __int64 v83; // [rsp+E8h] [rbp-48h]

  v57 = a11 + 72;
  v59 = *(_QWORD *)(a11 + 80);
  if ( v59 == a11 + 72 )
  {
    v40 = a1 + 40;
    v41 = a1 + 96;
    goto LABEL_74;
  }
  v56 = 0;
  do
  {
    if ( !v59 )
      BUG();
    v11 = *(_QWORD *)(v59 + 24);
    v12 = 0;
    v69 = v59 + 16;
    if ( v11 != v59 + 16 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v13 = v11;
          v11 = *(_QWORD *)(v11 + 8);
          v14 = *(_BYTE *)(v13 - 8);
          if ( v14 != 57 )
            break;
          v12 = 1;
          sub_15F20C0((_QWORD *)(v13 - 24));
LABEL_7:
          if ( v69 == v11 )
            goto LABEL_42;
        }
        if ( v14 != 58 )
        {
          if ( v14 == 59 )
          {
            v12 |= sub_19CC3C0(v13 - 24, a2, a3, a4, a5, a6, a7, a8, a9);
          }
          else if ( (v14 == 54 || v14 == 55) && sub_15F32D0(v13 - 24) )
          {
            *(_WORD *)(v13 - 6) &= 0xFC7Fu;
            *(_BYTE *)(v13 + 32) = 1;
          }
          goto LABEL_7;
        }
        v60 = (_QWORD *)(v13 - 24);
        v15 = sub_16498A0(v13 - 24);
        v78 = 0;
        v79 = v15;
        v80 = 0;
        v81 = 0;
        v82 = 0;
        v83 = 0;
        v16 = *(_QWORD *)(v13 + 16);
        v17 = *(unsigned __int8 **)(v13 + 24);
        v76 = 0;
        v77 = v16;
        v78 = (unsigned __int64 *)v13;
        v74[0] = v17;
        if ( v17 )
        {
          sub_1623A60((__int64)v74, (__int64)v17, 2);
          if ( v76 )
            sub_161E7C0((__int64)&v76, (__int64)v76);
          v76 = v74[0];
          if ( v74[0] )
            sub_1623210((__int64)v74, v74[0], (__int64)&v76);
        }
        v18 = *(_QWORD *)(v13 - 96);
        v67 = *(_QWORD *)(v13 - 72);
        v61 = *(_QWORD *)(v13 - 48);
        v75 = 257;
        v19 = sub_1648A60(64, 1u);
        if ( v19 )
          sub_15F9210((__int64)v19, *(_QWORD *)(*(_QWORD *)v18 + 24LL), v18, 0, 0, 0);
        if ( v77 )
        {
          v58 = v78;
          sub_157E9D0(v77 + 40, (__int64)v19);
          v20 = *v58;
          v21 = v19[3] & 7;
          v19[4] = (__int64)v58;
          v20 &= 0xFFFFFFFFFFFFFFF8LL;
          v19[3] = v20 | v21;
          *(_QWORD *)(v20 + 8) = v19 + 3;
          *v58 = *v58 & 7 | (unsigned __int64)(v19 + 3);
        }
        sub_164B780((__int64)v19, (__int64 *)v74);
        if ( v76 )
        {
          v72[0] = (__int64)v76;
          sub_1623A60((__int64)v72, (__int64)v76, 2);
          v22 = v19[6];
          v23 = (__int64)(v19 + 6);
          if ( v22 )
          {
            sub_161E7C0((__int64)(v19 + 6), v22);
            v23 = (__int64)(v19 + 6);
          }
          v24 = (unsigned __int8 *)v72[0];
          v19[6] = v72[0];
          if ( v24 )
            sub_1623210((__int64)v72, v24, v23);
        }
        v75 = 257;
        v68 = sub_12AA0C0((__int64 *)&v76, 0x20u, v19, v67, (__int64)v74);
        v75 = 257;
        v62 = sub_156B790((__int64 *)&v76, v68, v61, (__int64)v19, (__int64)v74, 0);
        v75 = 257;
        v25 = sub_1648A60(64, 2u);
        v26 = (__int64)v25;
        if ( v25 )
        {
          v27 = v62;
          v63 = v25;
          sub_15F9650((__int64)v25, v27, v18, 0, 0);
          v26 = (__int64)v63;
        }
        if ( v77 )
        {
          v28 = (__int64 *)v78;
          v64 = v26;
          sub_157E9D0(v77 + 40, v26);
          v26 = v64;
          v29 = *v28;
          v30 = *(_QWORD *)(v64 + 24);
          *(_QWORD *)(v64 + 32) = v28;
          v29 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v64 + 24) = v29 | v30 & 7;
          *(_QWORD *)(v29 + 8) = v64 + 24;
          *v28 = *v28 & 7 | (v64 + 24);
        }
        v65 = v26;
        sub_164B780(v26, (__int64 *)v74);
        if ( v76 )
        {
          v72[0] = (__int64)v76;
          sub_1623A60((__int64)v72, (__int64)v76, 2);
          v31 = v65;
          v32 = *(_QWORD *)(v65 + 48);
          if ( v32 )
          {
            sub_161E7C0(v65 + 48, v32);
            v31 = v65;
          }
          v33 = (unsigned __int8 *)v72[0];
          *(_QWORD *)(v31 + 48) = v72[0];
          if ( v33 )
            sub_1623210((__int64)v72, v33, v65 + 48);
        }
        v70 = 0;
        v73 = 257;
        v34 = sub_1599EF0(*(__int64 ***)(v13 - 24));
        v35 = (__int64 *)v34;
        if ( *(_BYTE *)(v34 + 16) > 0x10u || *((_BYTE *)v19 + 16) > 0x10u )
        {
          v75 = 257;
          v50 = sub_1648A60(88, 2u);
          v36 = (__int64)v50;
          if ( v50 )
          {
            v66 = (__int64)v50;
            sub_15F1EA0((__int64)v50, *v35, 63, (__int64)(v50 - 6), 2, 0);
            *(_QWORD *)(v36 + 56) = v36 + 72;
            *(_QWORD *)(v36 + 64) = 0x400000000LL;
            sub_15FAD90(v36, (__int64)v35, (__int64)v19, &v70, 1, (__int64)v74);
          }
          else
          {
            v66 = 0;
          }
          if ( v77 )
          {
            v51 = (__int64 *)v78;
            sub_157E9D0(v77 + 40, v36);
            v52 = *(_QWORD *)(v36 + 24);
            v53 = *v51;
            *(_QWORD *)(v36 + 32) = v51;
            v53 &= 0xFFFFFFFFFFFFFFF8LL;
            *(_QWORD *)(v36 + 24) = v53 | v52 & 7;
            *(_QWORD *)(v53 + 8) = v36 + 24;
            *v51 = *v51 & 7 | (v36 + 24);
          }
          sub_164B780(v66, v72);
          if ( v76 )
          {
            v71 = v76;
            sub_1623A60((__int64)&v71, (__int64)v76, 2);
            v54 = *(_QWORD *)(v36 + 48);
            if ( v54 )
              sub_161E7C0(v36 + 48, v54);
            v55 = v71;
            *(_QWORD *)(v36 + 48) = v71;
            if ( v55 )
              sub_1623210((__int64)&v71, v55, v36 + 48);
          }
        }
        else
        {
          v36 = sub_15A3A20((__int64 *)v34, v19, &v70, 1, 0);
        }
        v70 = 1;
        v73 = 257;
        if ( *(_BYTE *)(v36 + 16) > 0x10u || *(_BYTE *)(v68 + 16) > 0x10u )
        {
          v75 = 257;
          v43 = sub_1648A60(88, 2u);
          v37 = v43;
          if ( v43 )
          {
            v44 = (__int64)v43;
            sub_15F1EA0((__int64)v43, *(_QWORD *)v36, 63, (__int64)(v43 - 6), 2, 0);
            v37[7] = v37 + 9;
            v37[8] = 0x400000000LL;
            sub_15FAD90((__int64)v37, v36, v68, &v70, 1, (__int64)v74);
          }
          else
          {
            v44 = 0;
          }
          if ( v77 )
          {
            v45 = v78;
            sub_157E9D0(v77 + 40, (__int64)v37);
            v46 = v37[3];
            v47 = *v45;
            v37[4] = v45;
            v47 &= 0xFFFFFFFFFFFFFFF8LL;
            v37[3] = v47 | v46 & 7;
            *(_QWORD *)(v47 + 8) = v37 + 3;
            *v45 = *v45 & 7 | (unsigned __int64)(v37 + 3);
          }
          sub_164B780(v44, v72);
          if ( v76 )
          {
            v71 = v76;
            sub_1623A60((__int64)&v71, (__int64)v76, 2);
            v48 = v37[6];
            if ( v48 )
              sub_161E7C0((__int64)(v37 + 6), v48);
            v49 = v71;
            v37[6] = v71;
            if ( v49 )
              sub_1623210((__int64)&v71, v49, (__int64)(v37 + 6));
          }
        }
        else
        {
          v37 = (_QWORD *)sub_15A3A20((__int64 *)v36, (__int64 *)v68, &v70, 1, 0);
        }
        sub_164D160((__int64)v60, (__int64)v37, a2, a3, a4, a5, v38, v39, a8, a9);
        sub_15F20C0(v60);
        if ( v76 )
          sub_161E7C0((__int64)&v76, (__int64)v76);
        v12 = 1;
        if ( v69 == v11 )
        {
LABEL_42:
          v56 |= v12;
          break;
        }
      }
    }
    v59 = *(_QWORD *)(v59 + 8);
  }
  while ( v57 != v59 );
  v40 = a1 + 40;
  v41 = a1 + 96;
  if ( !v56 )
  {
LABEL_74:
    *(_QWORD *)(a1 + 24) = 0x100000002LL;
    *(_QWORD *)(a1 + 8) = v40;
    *(_QWORD *)(a1 + 16) = v40;
    *(_QWORD *)(a1 + 56) = 0;
    *(_QWORD *)(a1 + 64) = v41;
    *(_QWORD *)(a1 + 72) = v41;
    *(_QWORD *)(a1 + 80) = 2;
    *(_DWORD *)(a1 + 88) = 0;
    *(_DWORD *)(a1 + 32) = 0;
    *(_QWORD *)a1 = 1;
    *(_QWORD *)(a1 + 40) = &unk_4F9EE48;
    return a1;
  }
  memset((void *)a1, 0, 0x70u);
  *(_QWORD *)(a1 + 8) = v40;
  *(_QWORD *)(a1 + 16) = v40;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 64) = v41;
  *(_QWORD *)(a1 + 72) = v41;
  *(_DWORD *)(a1 + 80) = 2;
  return a1;
}
