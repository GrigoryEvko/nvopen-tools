// Function: sub_17FE640
// Address: 0x17fe640
//
__int64 __fastcall sub_17FE640(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  unsigned __int8 *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r12
  char v9; // r15
  unsigned int v10; // r14d
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned int v19; // ecx
  int v20; // ebx
  unsigned int v21; // r14d
  __int64 v22; // rsi
  int v23; // eax
  __int64 v24; // r13
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 *v27; // rbx
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // rsi
  unsigned __int8 *v31; // rsi
  unsigned int v32; // eax
  __int64 v33; // r13
  char v34; // al
  __int64 v35; // rsi
  __int64 v36; // rsi
  int v37; // eax
  __int64 v38; // rax
  _QWORD *v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 *v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rcx
  __int64 v45; // rsi
  unsigned __int8 *v46; // rsi
  __int64 *v47; // r15
  __int64 v48; // rax
  __int64 v49; // rcx
  __int64 v50; // rsi
  unsigned __int8 *v51; // rsi
  __int64 *v52; // r12
  __int64 v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rsi
  unsigned __int8 *v56; // rsi
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 **v59; // rdx
  _QWORD *v60; // rax
  _QWORD *v61; // r15
  unsigned __int64 *v62; // rbx
  __int64 v63; // rax
  unsigned __int64 v64; // rcx
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  __int64 *v67; // r15
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  int v72; // [rsp+Ch] [rbp-144h]
  __int64 v73; // [rsp+10h] [rbp-140h]
  unsigned __int64 v74; // [rsp+18h] [rbp-138h]
  int v75; // [rsp+20h] [rbp-130h]
  int v76; // [rsp+20h] [rbp-130h]
  int v77; // [rsp+20h] [rbp-130h]
  __int64 v78; // [rsp+20h] [rbp-130h]
  int v79; // [rsp+20h] [rbp-130h]
  int v80; // [rsp+20h] [rbp-130h]
  __int64 v81; // [rsp+20h] [rbp-130h]
  unsigned __int8 *v82; // [rsp+38h] [rbp-118h] BYREF
  _QWORD v83[2]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v84[2]; // [rsp+50h] [rbp-100h] BYREF
  __int16 v85; // [rsp+60h] [rbp-F0h]
  __int64 v86[2]; // [rsp+70h] [rbp-E0h] BYREF
  __int16 v87; // [rsp+80h] [rbp-D0h]
  __int64 v88[2]; // [rsp+90h] [rbp-C0h] BYREF
  __int16 v89; // [rsp+A0h] [rbp-B0h]
  unsigned __int8 *v90[2]; // [rsp+B0h] [rbp-A0h] BYREF
  __int16 v91; // [rsp+C0h] [rbp-90h]
  unsigned __int8 *v92; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v93; // [rsp+D8h] [rbp-78h]
  __int64 *v94; // [rsp+E0h] [rbp-70h]
  _QWORD *v95; // [rsp+E8h] [rbp-68h]
  __int64 v96; // [rsp+F0h] [rbp-60h]
  int v97; // [rsp+F8h] [rbp-58h]
  __int64 v98; // [rsp+100h] [rbp-50h]
  __int64 v99; // [rsp+108h] [rbp-48h]

  v5 = (_QWORD *)sub_16498A0(a2);
  v6 = *(unsigned __int8 **)(a2 + 48);
  v92 = 0;
  v95 = v5;
  v7 = *(_QWORD *)(a2 + 40);
  v96 = 0;
  v93 = v7;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v94 = (__int64 *)(a2 + 24);
  v90[0] = v6;
  if ( v6 )
  {
    sub_1623A60((__int64)v90, (__int64)v6, 2);
    if ( v92 )
      sub_161E7C0((__int64)&v92, (__int64)v92);
    v92 = v90[0];
    if ( v90[0] )
      sub_1623210((__int64)v90, v90[0], (__int64)&v92);
  }
  v8 = *(_QWORD *)(a2 - 24);
  v9 = *(_BYTE *)(a2 + 16);
  if ( (unsigned __int8)sub_1649A90(v8) || (v12 = sub_17FC080(*(_QWORD *)v8, a3), v13 = v12, v12 < 0) )
  {
    v10 = 0;
    goto LABEL_8;
  }
  v14 = *(_QWORD *)(a2 + 48);
  if ( v9 != 55 )
  {
    if ( v14 || (v19 = *(unsigned __int16 *)(a2 + 18), (v19 & 0x8000u) != 0) )
    {
      v75 = v13;
      v15 = sub_1625790(a2, 1);
      v13 = v75;
      if ( v15 )
      {
        LOBYTE(v16) = sub_14A7290(v15);
        v10 = v16;
        if ( (_BYTE)v16 )
        {
          v89 = 257;
          v87 = 257;
          v17 = sub_16471D0(v95, 0);
          if ( v17 != *(_QWORD *)v8 )
          {
            if ( *(_BYTE *)(v8 + 16) > 0x10u )
            {
              v91 = 257;
              v8 = sub_15FDFF0(v8, v17, (__int64)v90, 0);
              if ( v93 )
              {
                v42 = v94;
                sub_157E9D0(v93 + 40, v8);
                v43 = *(_QWORD *)(v8 + 24);
                v44 = *v42;
                *(_QWORD *)(v8 + 32) = v42;
                v44 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v8 + 24) = v44 | v43 & 7;
                *(_QWORD *)(v44 + 8) = v8 + 24;
                *v42 = *v42 & 7 | (v8 + 24);
              }
              sub_164B780(v8, v86);
              if ( v92 )
              {
                v84[0] = (__int64)v92;
                sub_1623A60((__int64)v84, (__int64)v92, 2);
                v45 = *(_QWORD *)(v8 + 48);
                if ( v45 )
                  sub_161E7C0(v8 + 48, v45);
                v46 = (unsigned __int8 *)v84[0];
                *(_QWORD *)(v8 + 48) = v84[0];
                if ( v46 )
                  sub_1623210((__int64)v84, v46, v8 + 48);
              }
            }
            else
            {
              v8 = sub_15A4A70((__int64 ***)v8, v17);
            }
          }
          v90[0] = (unsigned __int8 *)v8;
          sub_1285290(
            (__int64 *)&v92,
            *(_QWORD *)(*(_QWORD *)(a1 + 952) + 24LL),
            *(_QWORD *)(a1 + 952),
            (int)v90,
            1,
            (__int64)v88,
            0);
          goto LABEL_8;
        }
        v19 = *(unsigned __int16 *)(a2 + 18);
        v13 = v75;
        goto LABEL_23;
      }
      goto LABEL_22;
    }
LABEL_23:
    v20 = 1;
    v21 = 1 << (v19 >> 1) >> 1;
    v22 = *(_QWORD *)(*(_QWORD *)v8 + 24LL);
    while ( 2 )
    {
      switch ( *(_BYTE *)(v22 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v41 = *(_QWORD *)(v22 + 32);
          v22 = *(_QWORD *)(v22 + 24);
          v20 *= (_DWORD)v41;
          continue;
        case 1:
          v23 = 16;
          break;
        case 2:
          v23 = 32;
          break;
        case 3:
        case 9:
          v23 = 64;
          break;
        case 4:
          v23 = 80;
          break;
        case 5:
        case 6:
          v23 = 128;
          break;
        case 7:
          v80 = v13;
          v40 = sub_15A9520(a3, 0);
          v13 = v80;
          v23 = 8 * v40;
          break;
        case 0xB:
          v23 = *(_DWORD *)(v22 + 8) >> 8;
          break;
        case 0xD:
          v79 = v13;
          v39 = (_QWORD *)sub_15A9930(a3, v22);
          v13 = v79;
          v23 = 8 * *v39;
          break;
        case 0xE:
          v72 = v13;
          v73 = *(_QWORD *)(v22 + 24);
          v78 = *(_QWORD *)(v22 + 32);
          v74 = (unsigned int)sub_15A9FE0(a3, v73);
          v38 = sub_127FA20(a3, v73);
          v13 = v72;
          v23 = 8 * v78 * v74 * ((v74 + ((unsigned __int64)(v38 + 7) >> 3) - 1) / v74);
          break;
        case 0xF:
          v77 = v13;
          v37 = sub_15A9520(a3, *(_DWORD *)(v22 + 8) >> 8);
          v13 = v77;
          v23 = 8 * v37;
          break;
      }
      break;
    }
    if ( v21 - 1 <= 6 && v21 % ((unsigned int)(v23 * v20 + 7) >> 3) )
    {
      if ( v9 == 55 )
        v24 = *(_QWORD *)(a1 + 8 * v13 + 328);
      else
        v24 = *(_QWORD *)(a1 + 8 * v13 + 288);
    }
    else if ( v9 == 55 )
    {
      v24 = *(_QWORD *)(a1 + 8 * v13 + 248);
    }
    else
    {
      v24 = *(_QWORD *)(a1 + 8 * v13 + 208);
    }
    v87 = 257;
    v89 = 257;
    v25 = sub_16471D0(v95, 0);
    if ( v25 != *(_QWORD *)v8 )
    {
      if ( *(_BYTE *)(v8 + 16) > 0x10u )
      {
        v91 = 257;
        v26 = sub_15FDFF0(v8, v25, (__int64)v90, 0);
        v8 = v26;
        if ( v93 )
        {
          v27 = v94;
          sub_157E9D0(v93 + 40, v26);
          v28 = *(_QWORD *)(v8 + 24);
          v29 = *v27;
          *(_QWORD *)(v8 + 32) = v27;
          v29 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v8 + 24) = v29 | v28 & 7;
          *(_QWORD *)(v29 + 8) = v8 + 24;
          *v27 = *v27 & 7 | (v8 + 24);
        }
        sub_164B780(v8, v86);
        if ( v92 )
        {
          v84[0] = (__int64)v92;
          sub_1623A60((__int64)v84, (__int64)v92, 2);
          v30 = *(_QWORD *)(v8 + 48);
          if ( v30 )
            sub_161E7C0(v8 + 48, v30);
          v31 = (unsigned __int8 *)v84[0];
          *(_QWORD *)(v8 + 48) = v84[0];
          if ( v31 )
            sub_1623210((__int64)v84, v31, v8 + 48);
        }
      }
      else
      {
        v8 = sub_15A4A70((__int64 ***)v8, v25);
      }
    }
    v90[0] = (unsigned __int8 *)v8;
    v10 = 1;
    sub_1285290((__int64 *)&v92, *(_QWORD *)(*(_QWORD *)v24 + 24LL), v24, (int)v90, 1, (__int64)v88, 0);
    goto LABEL_8;
  }
  if ( !v14 )
  {
    v19 = *(unsigned __int16 *)(a2 + 18);
    if ( (v19 & 0x8000u) == 0 )
      goto LABEL_23;
  }
  v76 = v13;
  v18 = sub_1625790(a2, 1);
  v13 = v76;
  if ( !v18 || (LOBYTE(v32) = sub_14A7290(v18), v13 = v76, v10 = v32, !(_BYTE)v32) )
  {
LABEL_22:
    v19 = *(unsigned __int16 *)(a2 + 18);
    goto LABEL_23;
  }
  v33 = *(_QWORD *)(a2 - 48);
  v34 = *(_BYTE *)(*(_QWORD *)v33 + 8LL);
  if ( v34 == 16 )
  {
    v89 = 257;
    v57 = sub_1643350(v95);
    v58 = sub_159C470(v57, 0, 0);
    if ( *(_BYTE *)(v33 + 16) > 0x10u || *(_BYTE *)(v58 + 16) > 0x10u )
    {
      v81 = v58;
      v91 = 257;
      v60 = sub_1648A60(56, 2u);
      v61 = v60;
      if ( v60 )
        sub_15FA320((__int64)v60, (_QWORD *)v33, v81, (__int64)v90, 0);
      if ( v93 )
      {
        v62 = (unsigned __int64 *)v94;
        sub_157E9D0(v93 + 40, (__int64)v61);
        v63 = v61[3];
        v64 = *v62;
        v61[4] = v62;
        v64 &= 0xFFFFFFFFFFFFFFF8LL;
        v61[3] = v64 | v63 & 7;
        *(_QWORD *)(v64 + 8) = v61 + 3;
        *v62 = *v62 & 7 | (unsigned __int64)(v61 + 3);
      }
      sub_164B780((__int64)v61, v88);
      if ( v92 )
      {
        v86[0] = (__int64)v92;
        sub_1623A60((__int64)v86, (__int64)v92, 2);
        v65 = v61[6];
        if ( v65 )
          sub_161E7C0((__int64)(v61 + 6), v65);
        v66 = (unsigned __int8 *)v86[0];
        v61[6] = v86[0];
        if ( v66 )
          sub_1623210((__int64)v86, v66, (__int64)(v61 + 6));
      }
      v33 = (__int64)v61;
    }
    else
    {
      v33 = sub_15A37D0((_BYTE *)v33, v58, 0);
    }
    v34 = *(_BYTE *)(*(_QWORD *)v33 + 8LL);
  }
  if ( v34 == 11 )
  {
    v89 = 257;
    v59 = (__int64 **)sub_16471D0(v95, 0);
    if ( v59 != *(__int64 ***)v33 )
    {
      if ( *(_BYTE *)(v33 + 16) > 0x10u )
      {
        v91 = 257;
        v33 = sub_15FDBD0(46, v33, (__int64)v59, (__int64)v90, 0);
        if ( v93 )
        {
          v67 = v94;
          sub_157E9D0(v93 + 40, v33);
          v68 = *(_QWORD *)(v33 + 24);
          v69 = *v67;
          *(_QWORD *)(v33 + 32) = v67;
          v69 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v33 + 24) = v69 | v68 & 7;
          *(_QWORD *)(v69 + 8) = v33 + 24;
          *v67 = *v67 & 7 | (v33 + 24);
        }
        sub_164B780(v33, v88);
        if ( v92 )
        {
          v86[0] = (__int64)v92;
          sub_1623A60((__int64)v86, (__int64)v92, 2);
          v70 = *(_QWORD *)(v33 + 48);
          if ( v70 )
            sub_161E7C0(v33 + 48, v70);
          v71 = (unsigned __int8 *)v86[0];
          *(_QWORD *)(v33 + 48) = v86[0];
          if ( v71 )
            sub_1623210((__int64)v86, v71, v33 + 48);
        }
      }
      else
      {
        v33 = sub_15A46C0(46, (__int64 ***)v33, v59, 0);
      }
    }
  }
  v89 = 257;
  v85 = 257;
  v35 = sub_16471D0(v95, 0);
  if ( v35 != *(_QWORD *)v8 )
  {
    if ( *(_BYTE *)(v8 + 16) > 0x10u )
    {
      v91 = 257;
      v8 = sub_15FDFF0(v8, v35, (__int64)v90, 0);
      if ( v93 )
      {
        v47 = v94;
        sub_157E9D0(v93 + 40, v8);
        v48 = *(_QWORD *)(v8 + 24);
        v49 = *v47;
        *(_QWORD *)(v8 + 32) = v47;
        v49 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v8 + 24) = v49 | v48 & 7;
        *(_QWORD *)(v49 + 8) = v8 + 24;
        *v47 = *v47 & 7 | (v8 + 24);
      }
      sub_164B780(v8, v84);
      if ( v92 )
      {
        v86[0] = (__int64)v92;
        sub_1623A60((__int64)v86, (__int64)v92, 2);
        v50 = *(_QWORD *)(v8 + 48);
        if ( v50 )
          sub_161E7C0(v8 + 48, v50);
        v51 = (unsigned __int8 *)v86[0];
        *(_QWORD *)(v8 + 48) = v86[0];
        if ( v51 )
          sub_1623210((__int64)v86, v51, v8 + 48);
      }
    }
    else
    {
      v8 = sub_15A4A70((__int64 ***)v8, v35);
    }
  }
  v83[0] = v8;
  v87 = 257;
  v36 = sub_16471D0(v95, 0);
  if ( v36 != *(_QWORD *)v33 )
  {
    if ( *(_BYTE *)(v33 + 16) > 0x10u )
    {
      v91 = 257;
      v33 = sub_15FDFF0(v33, v36, (__int64)v90, 0);
      if ( v93 )
      {
        v52 = v94;
        sub_157E9D0(v93 + 40, v33);
        v53 = *(_QWORD *)(v33 + 24);
        v54 = *v52;
        *(_QWORD *)(v33 + 32) = v52;
        v54 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)(v33 + 24) = v54 | v53 & 7;
        *(_QWORD *)(v54 + 8) = v33 + 24;
        *v52 = *v52 & 7 | (v33 + 24);
      }
      sub_164B780(v33, v86);
      if ( v92 )
      {
        v82 = v92;
        sub_1623A60((__int64)&v82, (__int64)v92, 2);
        v55 = *(_QWORD *)(v33 + 48);
        if ( v55 )
          sub_161E7C0(v33 + 48, v55);
        v56 = v82;
        *(_QWORD *)(v33 + 48) = v82;
        if ( v56 )
          sub_1623210((__int64)&v82, v56, v33 + 48);
      }
    }
    else
    {
      v33 = sub_15A4A70((__int64 ***)v33, v36);
    }
  }
  v83[1] = v33;
  sub_1285290(
    (__int64 *)&v92,
    *(_QWORD *)(*(_QWORD *)(a1 + 944) + 24LL),
    *(_QWORD *)(a1 + 944),
    (int)v83,
    2,
    (__int64)v88,
    0);
LABEL_8:
  if ( v92 )
    sub_161E7C0((__int64)&v92, (__int64)v92);
  return v10;
}
