// Function: sub_1818700
// Address: 0x1818700
//
__int64 __fastcall sub_1818700(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        int a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        double a9)
{
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 result; // rax
  _QWORD *v14; // r15
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  unsigned __int64 *v17; // r15
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22; // r12
  __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rax
  __int64 *v26; // rax
  __int64 *v27; // r11
  __int64 v28; // rax
  unsigned __int64 *v29; // r12
  __int64 v30; // rax
  unsigned __int64 v31; // rcx
  __int64 v32; // rsi
  __int64 v33; // rdx
  unsigned __int8 *v34; // rsi
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // rax
  __int64 v38; // rax
  _QWORD *v39; // rax
  _QWORD *v40; // r14
  unsigned __int64 *v41; // r13
  __int64 v42; // rax
  unsigned __int64 v43; // rsi
  __int64 v44; // rsi
  __int64 v45; // rdx
  unsigned __int8 *v46; // rsi
  __int64 v47; // rax
  __int64 v48; // r12
  __int64 i; // rbx
  _QWORD *v50; // r13
  _QWORD *v51; // rax
  _QWORD *v52; // r15
  unsigned __int64 v53; // rsi
  __int64 v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdx
  unsigned __int8 *v57; // rsi
  __int64 v58; // rax
  _QWORD *v59; // rax
  unsigned __int64 *v60; // r15
  __int64 v61; // rax
  unsigned __int64 v62; // rsi
  __int64 v63; // rsi
  __int64 v64; // rdx
  unsigned __int8 *v65; // rsi
  __int64 v66; // rax
  __int64 v67; // rsi
  int v68; // r9d
  unsigned int v69; // edx
  __int64 *v70; // rbx
  __int64 v71; // r8
  __int64 *v72; // r12
  __int64 v73; // r13
  __int64 v74; // rax
  __int64 v75; // rax
  _QWORD *v76; // rax
  unsigned __int64 v77; // [rsp+0h] [rbp-140h]
  _BYTE *v78; // [rsp+10h] [rbp-130h]
  unsigned int v79; // [rsp+18h] [rbp-128h]
  unsigned __int64 *v80; // [rsp+20h] [rbp-120h]
  int v81; // [rsp+28h] [rbp-118h]
  __int64 v82; // [rsp+30h] [rbp-110h]
  __int64 v83; // [rsp+30h] [rbp-110h]
  __int64 v84; // [rsp+38h] [rbp-108h]
  unsigned __int64 v85; // [rsp+48h] [rbp-F8h]
  __int64 **v86; // [rsp+48h] [rbp-F8h]
  __int64 v87; // [rsp+50h] [rbp-F0h]
  __int64 *v90; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v91; // [rsp+78h] [rbp-C8h] BYREF
  __int64 v92[2]; // [rsp+80h] [rbp-C0h] BYREF
  __int16 v93; // [rsp+90h] [rbp-B0h]
  __int64 v94[2]; // [rsp+A0h] [rbp-A0h] BYREF
  __int16 v95; // [rsp+B0h] [rbp-90h]
  __int64 v96; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v97; // [rsp+C8h] [rbp-78h]
  unsigned __int64 *v98; // [rsp+D0h] [rbp-70h]
  _QWORD *v99; // [rsp+D8h] [rbp-68h]

  if ( *(_BYTE *)(a2 + 16) == 53 )
  {
    v66 = *(unsigned int *)(a1 + 184);
    if ( (_DWORD)v66 )
    {
      v67 = *(_QWORD *)(a1 + 168);
      v68 = 1;
      v69 = (v66 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v70 = (__int64 *)(v67 + 16LL * v69);
      v71 = *v70;
      if ( a2 == *v70 )
      {
LABEL_78:
        if ( v70 != (__int64 *)(v67 + 16 * v66) )
        {
          sub_17CE510((__int64)&v96, a6, 0, 0, 0);
          sub_12A8F50(&v96, a5, v70[1], 0);
          return sub_17CD270(&v96);
        }
      }
      else
      {
        while ( v71 != -8 )
        {
          v69 = (v66 - 1) & (v68 + v69);
          v70 = (__int64 *)(v67 + 16LL * v69);
          v71 = *v70;
          if ( a2 == *v70 )
            goto LABEL_78;
          ++v68;
        }
      }
    }
  }
  v79 = 2 * a4;
  sub_17CE510((__int64)&v96, a6, 0, 0, 0);
  v78 = (_BYTE *)sub_18165C0(*(_QWORD *)a1, a2, a6, a7, a8, a9);
  v11 = *(_QWORD *)a1;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 200LL) == a5 )
  {
    v72 = (__int64 *)sub_1644900(*(_QWORD **)(v11 + 168), 16 * (int)a3);
    v73 = sub_159C470((__int64)v72, 0, 0);
    v95 = 257;
    v74 = sub_1646BA0(v72, 0);
    v75 = sub_12AA3B0(&v96, 0x2Fu, (__int64)v78, v74, (__int64)v94);
    v76 = sub_12A8F50(&v96, v73, v75, 0);
    sub_15F9450((__int64)v76, v79);
    return sub_17CD270(&v96);
  }
  else
  {
    v12 = 0;
    if ( a3 > 7 )
    {
      v86 = (__int64 **)sub_16463B0(*(__int64 **)(v11 + 176), 8u);
      v36 = sub_1599EF0(v86);
      do
      {
        while ( 1 )
        {
          v93 = 257;
          v37 = sub_1643350(*(_QWORD **)(*(_QWORD *)a1 + 168LL));
          v38 = sub_159C470(v37, v12, 0);
          if ( *(_BYTE *)(v36 + 16) > 0x10u || *(_BYTE *)(a5 + 16) > 0x10u || *(_BYTE *)(v38 + 16) > 0x10u )
            break;
          ++v12;
          v36 = sub_15A3890((__int64 *)v36, a5, v38, 0);
          if ( v12 == 8 )
            goto LABEL_54;
        }
        v87 = v38;
        v95 = 257;
        v39 = sub_1648A60(56, 3u);
        v40 = v39;
        if ( v39 )
          sub_15FA480((__int64)v39, (__int64 *)v36, a5, v87, (__int64)v94, 0);
        if ( v97 )
        {
          v41 = v98;
          sub_157E9D0(v97 + 40, (__int64)v40);
          v42 = v40[3];
          v43 = *v41;
          v40[4] = v41;
          v43 &= 0xFFFFFFFFFFFFFFF8LL;
          v40[3] = v43 | v42 & 7;
          *(_QWORD *)(v43 + 8) = v40 + 3;
          *v41 = *v41 & 7 | (unsigned __int64)(v40 + 3);
        }
        sub_164B780((__int64)v40, v92);
        if ( v96 )
        {
          v91 = v96;
          sub_1623A60((__int64)&v91, v96, 2);
          v44 = v40[6];
          v45 = (__int64)(v40 + 6);
          if ( v44 )
          {
            sub_161E7C0((__int64)(v40 + 6), v44);
            v45 = (__int64)(v40 + 6);
          }
          v46 = (unsigned __int8 *)v91;
          v40[6] = v91;
          if ( v46 )
            sub_1623210((__int64)&v91, v46, v45);
        }
        ++v12;
        v36 = (__int64)v40;
      }
      while ( v12 != 8 );
LABEL_54:
      v95 = 257;
      v47 = sub_1646BA0((__int64 *)v86, 0);
      v83 = v36;
      v48 = sub_12AA3B0(&v96, 0x2Fu, (__int64)v78, v47, (__int64)v94);
      v77 = ((a3 - 8) >> 3) + 1;
      for ( i = 0; ; ++i )
      {
        v93 = 257;
        v58 = sub_1643350(v99);
        v90 = (__int64 *)sub_159C470(v58, (unsigned int)i, 0);
        if ( *(_BYTE *)(v48 + 16) <= 0x10u )
        {
          BYTE4(v94[0]) = 0;
          v50 = (_QWORD *)sub_15A2E80((__int64)v86, v48, &v90, 1u, 0, (__int64)v94, 0);
        }
        else
        {
          v95 = 257;
          v59 = sub_1704C00((__int64)v86, v48, (__int64 *)&v90, 1, (__int64)v94, 0);
          v50 = v59;
          if ( v97 )
          {
            v60 = v98;
            sub_157E9D0(v97 + 40, (__int64)v59);
            v61 = v50[3];
            v62 = *v60;
            v50[4] = v60;
            v62 &= 0xFFFFFFFFFFFFFFF8LL;
            v50[3] = v62 | v61 & 7;
            *(_QWORD *)(v62 + 8) = v50 + 3;
            *v60 = *v60 & 7 | (unsigned __int64)(v50 + 3);
          }
          sub_164B780((__int64)v50, v92);
          if ( v96 )
          {
            v91 = v96;
            sub_1623A60((__int64)&v91, v96, 2);
            v63 = v50[6];
            v64 = (__int64)(v50 + 6);
            if ( v63 )
            {
              sub_161E7C0((__int64)(v50 + 6), v63);
              v64 = (__int64)(v50 + 6);
            }
            v65 = (unsigned __int8 *)v91;
            v50[6] = v91;
            if ( v65 )
              sub_1623210((__int64)&v91, v65, v64);
          }
        }
        v95 = 257;
        v51 = sub_1648A60(64, 2u);
        v52 = v51;
        if ( v51 )
          sub_15F9650((__int64)v51, v83, (__int64)v50, 0, 0);
        if ( v97 )
        {
          v80 = v98;
          sub_157E9D0(v97 + 40, (__int64)v52);
          v53 = *v80;
          v54 = v52[3] & 7LL;
          v52[4] = v80;
          v53 &= 0xFFFFFFFFFFFFFFF8LL;
          v52[3] = v53 | v54;
          *(_QWORD *)(v53 + 8) = v52 + 3;
          *v80 = *v80 & 7 | (unsigned __int64)(v52 + 3);
        }
        sub_164B780((__int64)v52, v94);
        if ( v96 )
        {
          v92[0] = v96;
          sub_1623A60((__int64)v92, v96, 2);
          v55 = v52[6];
          v56 = (__int64)(v52 + 6);
          if ( v55 )
          {
            sub_161E7C0((__int64)(v52 + 6), v55);
            v56 = (__int64)(v52 + 6);
          }
          v57 = (unsigned __int8 *)v92[0];
          v52[6] = v92[0];
          if ( v57 )
            sub_1623210((__int64)v92, v57, v56);
        }
        sub_15F9450((__int64)v52, v79);
        if ( i == (a3 - 8) >> 3 )
          break;
      }
      a3 &= 7u;
      v12 = 8 * v77;
    }
    result = a3;
    v85 = a3 + v12;
    if ( a3 )
    {
      do
      {
        v93 = 257;
        v22 = *(_QWORD *)(*(_QWORD *)a1 + 176LL);
        v23 = sub_1643350(v99);
        v90 = (__int64 *)sub_159C470(v23, (unsigned int)v12, 0);
        if ( v78[16] <= 0x10u )
        {
          BYTE4(v94[0]) = 0;
          v14 = (_QWORD *)sub_15A2E80(v22, (__int64)v78, &v90, 1u, 0, (__int64)v94, 0);
        }
        else
        {
          v95 = 257;
          if ( !v22 )
          {
            v35 = *(_QWORD *)v78;
            if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) == 16 )
              v35 = **(_QWORD **)(v35 + 16);
            v22 = *(_QWORD *)(v35 + 24);
          }
          v24 = sub_1648A60(72, 2u);
          v14 = v24;
          if ( v24 )
          {
            v84 = (__int64)v24;
            v82 = (__int64)(v24 - 6);
            v25 = *(_QWORD *)v78;
            if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) == 16 )
              v25 = **(_QWORD **)(v25 + 16);
            v81 = *(_DWORD *)(v25 + 8) >> 8;
            v26 = (__int64 *)sub_15F9F50(v22, (__int64)&v90, 1);
            v27 = (__int64 *)sub_1646BA0(v26, v81);
            v28 = *(_QWORD *)v78;
            if ( *(_BYTE *)(*(_QWORD *)v78 + 8LL) == 16 || (v28 = *v90, *(_BYTE *)(*v90 + 8) == 16) )
              v27 = sub_16463B0(v27, *(_QWORD *)(v28 + 32));
            sub_15F1EA0((__int64)v14, (__int64)v27, 32, v82, 2, 0);
            v14[7] = v22;
            v14[8] = sub_15F9F50(v22, (__int64)&v90, 1);
            sub_15F9CE0((__int64)v14, (__int64)v78, (__int64 *)&v90, 1, (__int64)v94);
          }
          else
          {
            v84 = 0;
          }
          if ( v97 )
          {
            v29 = v98;
            sub_157E9D0(v97 + 40, (__int64)v14);
            v30 = v14[3];
            v31 = *v29;
            v14[4] = v29;
            v31 &= 0xFFFFFFFFFFFFFFF8LL;
            v14[3] = v31 | v30 & 7;
            *(_QWORD *)(v31 + 8) = v14 + 3;
            *v29 = *v29 & 7 | (unsigned __int64)(v14 + 3);
          }
          sub_164B780(v84, v92);
          if ( v96 )
          {
            v91 = v96;
            sub_1623A60((__int64)&v91, v96, 2);
            v32 = v14[6];
            v33 = (__int64)(v14 + 6);
            if ( v32 )
            {
              sub_161E7C0((__int64)(v14 + 6), v32);
              v33 = (__int64)(v14 + 6);
            }
            v34 = (unsigned __int8 *)v91;
            v14[6] = v91;
            if ( v34 )
              sub_1623210((__int64)&v91, v34, v33);
          }
        }
        v95 = 257;
        v15 = sub_1648A60(64, 2u);
        v16 = v15;
        if ( v15 )
          sub_15F9650((__int64)v15, a5, (__int64)v14, 0, 0);
        if ( v97 )
        {
          v17 = v98;
          sub_157E9D0(v97 + 40, (__int64)v16);
          v18 = v16[3];
          v19 = *v17;
          v16[4] = v17;
          v19 &= 0xFFFFFFFFFFFFFFF8LL;
          v16[3] = v19 | v18 & 7;
          *(_QWORD *)(v19 + 8) = v16 + 3;
          *v17 = *v17 & 7 | (unsigned __int64)(v16 + 3);
        }
        sub_164B780((__int64)v16, v94);
        if ( v96 )
        {
          v92[0] = v96;
          sub_1623A60((__int64)v92, v96, 2);
          v20 = v16[6];
          if ( v20 )
            sub_161E7C0((__int64)(v16 + 6), v20);
          v21 = (unsigned __int8 *)v92[0];
          v16[6] = v92[0];
          if ( v21 )
            sub_1623210((__int64)v92, v21, (__int64)(v16 + 6));
        }
        ++v12;
        result = sub_15F9450((__int64)v16, v79);
      }
      while ( v12 != v85 );
    }
    if ( v96 )
      return sub_161E7C0((__int64)&v96, v96);
  }
  return result;
}
