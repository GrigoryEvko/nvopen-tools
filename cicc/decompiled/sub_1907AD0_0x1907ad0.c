// Function: sub_1907AD0
// Address: 0x1907ad0
//
_QWORD *__fastcall sub_1907AD0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r15
  __int64 v13; // r13
  __int64 v15; // rax
  __int64 v16; // rsi
  int v17; // r9d
  __int64 v18; // r11
  __int64 *v19; // rcx
  __int64 v20; // rdi
  _QWORD *v21; // rbx
  __int64 *v23; // r8
  __int64 *v24; // rbx
  __int64 v25; // r9
  int v26; // r9d
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int8 *v30; // rsi
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // r9
  int v34; // r9d
  __int64 v35; // r15
  __int64 v36; // rax
  unsigned int v37; // esi
  __int64 v38; // rdx
  int v39; // eax
  double v40; // xmm4_8
  double v41; // xmm5_8
  _QWORD *v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // r15
  __int64 v45; // rax
  int v46; // eax
  int v47; // r12d
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r15
  __int64 v51; // rdx
  __int64 v52; // rcx
  _QWORD *v53; // rax
  __int64 v54; // r9
  _QWORD **v55; // rax
  __int64 *v56; // rax
  __int64 v57; // rax
  __int64 v58; // r9
  unsigned __int64 *v59; // r12
  __int64 v60; // rax
  unsigned __int64 v61; // rcx
  __int64 v62; // rsi
  unsigned __int8 *v63; // rsi
  _QWORD *v64; // rdx
  __int64 v65; // [rsp+8h] [rbp-138h]
  __int64 v66; // [rsp+8h] [rbp-138h]
  __int64 v67; // [rsp+8h] [rbp-138h]
  _QWORD *v68; // [rsp+10h] [rbp-130h]
  __int64 v69; // [rsp+10h] [rbp-130h]
  __int64 *v70; // [rsp+20h] [rbp-120h]
  __int64 v71; // [rsp+20h] [rbp-120h]
  __int64 v72; // [rsp+20h] [rbp-120h]
  unsigned __int64 v73[2]; // [rsp+28h] [rbp-118h] BYREF
  unsigned __int8 *v74; // [rsp+38h] [rbp-108h] BYREF
  _QWORD v75[2]; // [rsp+40h] [rbp-100h] BYREF
  __int64 v76[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v77; // [rsp+60h] [rbp-E0h]
  unsigned __int8 *v78[2]; // [rsp+70h] [rbp-D0h] BYREF
  __int16 v79; // [rsp+80h] [rbp-C0h]
  __int64 *v80; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v81; // [rsp+98h] [rbp-A8h]
  _BYTE v82[32]; // [rsp+A0h] [rbp-A0h] BYREF
  unsigned __int8 *v83; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v84; // [rsp+C8h] [rbp-78h]
  unsigned __int64 *v85; // [rsp+D0h] [rbp-70h]
  __int64 v86; // [rsp+D8h] [rbp-68h]
  __int64 v87; // [rsp+E0h] [rbp-60h]
  int v88; // [rsp+E8h] [rbp-58h]
  __int64 v89; // [rsp+F0h] [rbp-50h]
  __int64 v90; // [rsp+F8h] [rbp-48h]

  v11 = a2;
  v13 = a1 + 208;
  v15 = *(unsigned int *)(a1 + 232);
  v73[0] = a2;
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD *)(a1 + 216);
    v17 = 1;
    LODWORD(v18) = (v15 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v19 = (__int64 *)(v16 + 16LL * (unsigned int)v18);
    v20 = *v19;
    if ( v11 == *v19 )
    {
LABEL_3:
      if ( v19 != (__int64 *)(v16 + 16 * v15)
        && *(_QWORD *)(a1 + 248) != *(_QWORD *)(a1 + 240) + 16LL * *((unsigned int *)v19 + 2) )
      {
        return *(_QWORD **)sub_1907820(v13, v73);
      }
    }
    else
    {
      while ( v20 != -8 )
      {
        v18 = ((_DWORD)v15 - 1) & (unsigned int)(v18 + v17);
        v19 = (__int64 *)(v16 + 16 * v18);
        v20 = *v19;
        if ( v11 == *v19 )
          goto LABEL_3;
        ++v17;
      }
    }
  }
  v80 = (__int64 *)v82;
  v81 = 0x400000000LL;
  if ( (*(_BYTE *)(v11 + 23) & 0x40) != 0 )
  {
    v23 = *(__int64 **)(v11 - 8);
    v70 = &v23[3 * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF)];
  }
  else
  {
    v70 = (__int64 *)v11;
    v23 = (__int64 *)(v11 - 24LL * (*(_DWORD *)(v11 + 20) & 0xFFFFFFF));
  }
  if ( v23 != v70 )
  {
    v24 = v23;
    do
    {
      v25 = *v24;
      if ( (unsigned __int8)(*(_BYTE *)(v11 + 16) - 65) <= 1u )
      {
        v31 = (unsigned int)v81;
        if ( (unsigned int)v81 >= HIDWORD(v81) )
        {
          v66 = *v24;
          sub_16CD150((__int64)&v80, v82, 0, 8, (int)v23, v25);
          v31 = (unsigned int)v81;
          v25 = v66;
        }
        v80[v31] = v25;
        LODWORD(v81) = v81 + 1;
      }
      else if ( *(_BYTE *)(v25 + 16) <= 0x17u )
      {
        v65 = *v24;
        v32 = sub_1643030(a3);
        v33 = v65;
        LODWORD(v84) = v32;
        if ( v32 <= 0x40 )
        {
          v83 = 0;
        }
        else
        {
          sub_16A4EF0((__int64)&v83, 0, 0);
          v33 = v65;
        }
        BYTE4(v84) = 0;
        sub_169E1A0(v33 + 24, (__int64)&v83, 0, v78);
        v35 = sub_15A1070(a3, (__int64)&v83);
        v36 = (unsigned int)v81;
        if ( (unsigned int)v81 >= HIDWORD(v81) )
        {
          sub_16CD150((__int64)&v80, v82, 0, 8, (int)v23, v34);
          v36 = (unsigned int)v81;
        }
        v80[v36] = v35;
        LODWORD(v81) = v81 + 1;
        if ( (unsigned int)v84 > 0x40 && v83 )
          j_j___libc_free_0_0(v83);
      }
      else
      {
        v27 = sub_1907AD0(a1, *v24, a3);
        v28 = (unsigned int)v81;
        if ( (unsigned int)v81 >= HIDWORD(v81) )
        {
          sub_16CD150((__int64)&v80, v82, 0, 8, (int)v23, v26);
          v28 = (unsigned int)v81;
        }
        v80[v28] = v27;
        LODWORD(v81) = v81 + 1;
      }
      v11 = v73[0];
      v24 += 3;
    }
    while ( v70 != v24 );
  }
  v29 = sub_16498A0(v11);
  v83 = 0;
  v86 = v29;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v84 = *(_QWORD *)(v11 + 40);
  v85 = (unsigned __int64 *)(v11 + 24);
  v30 = *(unsigned __int8 **)(v11 + 48);
  v78[0] = v30;
  if ( v30 )
  {
    sub_1623A60((__int64)v78, (__int64)v30, 2);
    if ( v83 )
      sub_161E7C0((__int64)&v83, (__int64)v83);
    v83 = v78[0];
    if ( v78[0] )
      sub_1623210((__int64)v78, v78[0], (__int64)&v83);
  }
  switch ( *(_BYTE *)(v73[0] + 16) )
  {
    case '$':
    case '&':
    case '(':
      v37 = 13;
      v76[0] = (__int64)sub_1649960(v73[0]);
      v79 = 261;
      v78[0] = (unsigned __int8 *)v76;
      v76[1] = v38;
      v39 = *(unsigned __int8 *)(v73[0] + 16);
      if ( v39 != 38 )
        v37 = 4 * (v39 == 40) + 11;
      v21 = (_QWORD *)sub_1904E90((__int64)&v83, v37, *v80, v80[1], (__int64 *)v78, 0, *(double *)a4.m128_u64, a5, a6);
      break;
    case '?':
      v79 = 257;
      v21 = (_QWORD *)sub_1904B50((__int64 *)&v83, *v80, *(_QWORD *)v73[0], (__int64 *)v78);
      break;
    case '@':
      v79 = 257;
      v21 = (_QWORD *)sub_1904CF0((__int64 *)&v83, *v80, *(_QWORD *)v73[0], (__int64 *)v78);
      break;
    case 'A':
      v79 = 257;
      v21 = (_QWORD *)sub_1904B50((__int64 *)&v83, *v80, a3, (__int64 *)v78);
      break;
    case 'B':
      v79 = 257;
      v21 = (_QWORD *)sub_1904CF0((__int64 *)&v83, *v80, a3, (__int64 *)v78);
      break;
    case 'L':
      v46 = *(unsigned __int16 *)(v73[0] + 18);
      LOWORD(v47) = 42;
      BYTE1(v46) &= ~0x80u;
      v48 = (unsigned int)(v46 - 1);
      if ( (unsigned int)v48 <= 0xD )
        v47 = *(_DWORD *)&asc_42BDC60[4 * v48];
      v75[0] = sub_1649960(v73[0]);
      v76[0] = (__int64)v75;
      v75[1] = v49;
      v77 = 261;
      v50 = *v80;
      if ( *(_BYTE *)(*v80 + 16) > 0x10u || *(_BYTE *)(v80[1] + 16) > 0x10u )
      {
        v71 = v80[1];
        v79 = 257;
        v53 = sub_1648A60(56, 2u);
        v54 = v71;
        v21 = v53;
        if ( v53 )
        {
          v72 = (__int64)v53;
          v55 = *(_QWORD ***)v50;
          if ( *(_BYTE *)(*(_QWORD *)v50 + 8LL) == 16 )
          {
            v67 = v54;
            v68 = v55[4];
            v56 = (__int64 *)sub_1643320(*v55);
            v57 = (__int64)sub_16463B0(v56, (unsigned int)v68);
            v58 = v67;
          }
          else
          {
            v69 = v54;
            v57 = sub_1643320(*v55);
            v58 = v69;
          }
          sub_15FEC10((__int64)v21, v57, 51, v47, v50, v58, (__int64)v78, 0);
        }
        else
        {
          v72 = 0;
        }
        if ( v84 )
        {
          v59 = v85;
          sub_157E9D0(v84 + 40, (__int64)v21);
          v60 = v21[3];
          v61 = *v59;
          v21[4] = v59;
          v61 &= 0xFFFFFFFFFFFFFFF8LL;
          v21[3] = v61 | v60 & 7;
          *(_QWORD *)(v61 + 8) = v21 + 3;
          *v59 = *v59 & 7 | (unsigned __int64)(v21 + 3);
        }
        sub_164B780(v72, v76);
        if ( v83 )
        {
          v74 = v83;
          sub_1623A60((__int64)&v74, (__int64)v83, 2);
          v62 = v21[6];
          if ( v62 )
            sub_161E7C0((__int64)(v21 + 6), v62);
          v63 = v74;
          v21[6] = v74;
          if ( v63 )
            sub_1623210((__int64)&v74, v63, (__int64)(v21 + 6));
        }
      }
      else
      {
        v21 = (_QWORD *)sub_15A37B0(v47, (_QWORD *)*v80, (_QWORD *)v80[1], 0);
      }
      break;
    default:
      BUG();
  }
  v42 = *(_QWORD **)(a1 + 72);
  v43 = *(_QWORD **)(a1 + 64);
  if ( v42 == v43 )
  {
    v44 = &v43[*(unsigned int *)(a1 + 84)];
    if ( v43 == v44 )
    {
      v64 = *(_QWORD **)(a1 + 64);
    }
    else
    {
      do
      {
        if ( v73[0] == *v43 )
          break;
        ++v43;
      }
      while ( v44 != v43 );
      v64 = v44;
    }
LABEL_65:
    while ( v64 != v43 )
    {
      if ( *v43 < 0xFFFFFFFFFFFFFFFELL )
        break;
      ++v43;
    }
    goto LABEL_45;
  }
  v44 = &v42[*(unsigned int *)(a1 + 80)];
  v43 = sub_16CC9F0(a1 + 56, v73[0]);
  if ( v73[0] == *v43 )
  {
    v51 = *(_QWORD *)(a1 + 72);
    if ( v51 == *(_QWORD *)(a1 + 64) )
      v52 = *(unsigned int *)(a1 + 84);
    else
      v52 = *(unsigned int *)(a1 + 80);
    v64 = (_QWORD *)(v51 + 8 * v52);
    goto LABEL_65;
  }
  v45 = *(_QWORD *)(a1 + 72);
  if ( v45 == *(_QWORD *)(a1 + 64) )
  {
    v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(a1 + 84));
    v64 = v43;
    goto LABEL_65;
  }
  v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_45:
  if ( v43 != v44 )
    sub_164D160(v73[0], (__int64)v21, a4, a5, a6, a7, v40, v41, a10, a11);
  *(_QWORD *)sub_1907820(v13, v73) = v21;
  if ( v83 )
    sub_161E7C0((__int64)&v83, (__int64)v83);
  if ( v80 != (__int64 *)v82 )
    _libc_free((unsigned __int64)v80);
  return v21;
}
