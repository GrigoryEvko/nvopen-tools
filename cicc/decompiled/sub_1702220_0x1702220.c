// Function: sub_1702220
// Address: 0x1702220
//
__int64 __fastcall sub_1702220(
        __int64 a1,
        __int64 *a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // rbx
  __int64 v12; // r14
  __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  unsigned __int8 *v16; // rsi
  __int64 *v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r10
  _QWORD *v24; // rax
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 v27; // r13
  __int64 v28; // r14
  __int64 **v29; // r15
  __int64 v30; // rax
  unsigned __int8 *v31; // rsi
  __int64 result; // rax
  __int64 v33; // rbx
  __int64 i; // r12
  _QWORD *v35; // rdi
  __int64 **v36; // rsi
  __int64 ****v37; // rax
  __int64 ***v38; // rdx
  char v39; // dl
  __int64 *v40; // rdi
  __int64 v41; // rdx
  __int64 *v42; // rcx
  __int64 v43; // rsi
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // r10
  char v48; // al
  int v49; // r15d
  __int64 *v50; // r15
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // r15
  unsigned __int8 *v55; // rsi
  __int64 ****v56; // rax
  __int64 ***v57; // rdi
  __int64 v58; // r10
  __int64 *v59; // r13
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rsi
  __int64 v63; // r15
  unsigned __int8 *v64; // rsi
  __int64 *v65; // rbx
  __int64 v66; // rax
  __int64 v67; // rcx
  __int64 v68; // rsi
  unsigned __int8 *v69; // rsi
  __int64 v70; // [rsp+0h] [rbp-F0h]
  __int64 *v71; // [rsp+8h] [rbp-E8h]
  unsigned int v72; // [rsp+10h] [rbp-E0h]
  __int64 v73; // [rsp+10h] [rbp-E0h]
  __int64 v74; // [rsp+10h] [rbp-E0h]
  __int64 v75; // [rsp+10h] [rbp-E0h]
  __int64 v76; // [rsp+10h] [rbp-E0h]
  __int64 v77; // [rsp+10h] [rbp-E0h]
  __int64 v78; // [rsp+10h] [rbp-E0h]
  __int64 v79; // [rsp+10h] [rbp-E0h]
  __int64 v80; // [rsp+10h] [rbp-E0h]
  __int64 v81; // [rsp+10h] [rbp-E0h]
  __int64 v82; // [rsp+10h] [rbp-E0h]
  unsigned __int8 *v84; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v85[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v86; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v87[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v88; // [rsp+60h] [rbp-90h]
  unsigned __int8 *v89; // [rsp+70h] [rbp-80h] BYREF
  __int64 v90; // [rsp+78h] [rbp-78h]
  __int64 *v91; // [rsp+80h] [rbp-70h]
  __int64 v92; // [rsp+88h] [rbp-68h]
  __int64 v93; // [rsp+90h] [rbp-60h]
  int v94; // [rsp+98h] [rbp-58h]
  __int64 v95; // [rsp+A0h] [rbp-50h]
  __int64 v96; // [rsp+A8h] [rbp-48h]

  v11 = *(__int64 **)(a1 + 112);
  v71 = *(__int64 **)(a1 + 120);
  if ( v11 != v71 )
  {
    while ( 2 )
    {
      v12 = *v11;
      v13 = sub_16498A0(*v11);
      v89 = 0;
      v92 = v13;
      v93 = 0;
      v94 = 0;
      v95 = 0;
      v96 = 0;
      v90 = *(_QWORD *)(v12 + 40);
      v91 = (__int64 *)(v12 + 24);
      v16 = *(unsigned __int8 **)(v12 + 48);
      v87[0] = v16;
      if ( v16 )
      {
        sub_1623A60((__int64)v87, (__int64)v16, 2);
        if ( v89 )
          sub_161E7C0((__int64)&v89, (__int64)v89);
        v89 = v87[0];
        if ( v87[0] )
          sub_1623210((__int64)v87, v87[0], (__int64)&v89);
      }
      v72 = *(unsigned __int8 *)(v12 + 16) - 24;
      switch ( *(_BYTE *)(v12 + 16) )
      {
        case '#':
        case '%':
        case '\'':
        case '2':
        case '3':
        case '4':
          if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
            v17 = *(__int64 **)(v12 - 8);
          else
            v17 = (__int64 *)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
          v18 = sub_1702130(a1, *v17, a2);
          if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
            v19 = *(_QWORD *)(v12 - 8);
          else
            v19 = v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF);
          v20 = sub_1702130(a1, *(_QWORD *)(v19 + 24), a2);
          v86 = 257;
          v21 = v20;
          if ( *(_BYTE *)(v18 + 16) > 0x10u
            || *(_BYTE *)(v20 + 16) > 0x10u
            || (v70 = v20,
                v22 = sub_15A2A30((__int64 *)v72, (__int64 *)v18, v20, 0, 0, *(double *)a3.m128_u64, a4, a5),
                v21 = v70,
                (v23 = v22) == 0) )
          {
            v88 = 257;
            v45 = sub_15FB440(v72, (__int64 *)v18, v21, (__int64)v87, 0);
            v46 = *(_QWORD *)v45;
            v47 = v45;
            v48 = *(_BYTE *)(*(_QWORD *)v45 + 8LL);
            if ( v48 == 16 )
              v48 = *(_BYTE *)(**(_QWORD **)(v46 + 16) + 8LL);
            if ( (unsigned __int8)(v48 - 1) <= 5u || *(_BYTE *)(v47 + 16) == 76 )
            {
              v49 = v94;
              if ( v93 )
              {
                v74 = v47;
                sub_1625C10(v47, 3, v93);
                v47 = v74;
              }
              v75 = v47;
              sub_15F2440(v47, v49);
              v47 = v75;
            }
            if ( v90 )
            {
              v50 = v91;
              v76 = v47;
              sub_157E9D0(v90 + 40, v47);
              v47 = v76;
              v51 = *v50;
              v52 = *(_QWORD *)(v76 + 24);
              *(_QWORD *)(v76 + 32) = v50;
              v51 &= 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)(v76 + 24) = v51 | v52 & 7;
              *(_QWORD *)(v51 + 8) = v76 + 24;
              *v50 = *v50 & 7 | (v76 + 24);
            }
            v77 = v47;
            sub_164B780(v47, v85);
            v23 = v77;
            if ( v89 )
            {
              v87[0] = v89;
              sub_1623A60((__int64)v87, (__int64)v89, 2);
              v23 = v77;
              v53 = *(_QWORD *)(v77 + 48);
              v54 = v77 + 48;
              if ( v53 )
              {
                sub_161E7C0(v77 + 48, v53);
                v23 = v77;
              }
              v55 = v87[0];
              *(unsigned __int8 **)(v23 + 48) = v87[0];
              if ( v55 )
              {
                v78 = v23;
                sub_1623210((__int64)v87, v55, v54);
                v23 = v78;
              }
            }
          }
          goto LABEL_15;
        case '<':
        case '=':
        case '>':
          v36 = (__int64 **)a2;
          if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) == 16 )
            v36 = (__int64 **)sub_16463B0(a2, *(_QWORD *)(*(_QWORD *)v12 + 32LL));
          if ( (*(_BYTE *)(v12 + 23) & 0x40) != 0 )
          {
            v56 = *(__int64 *****)(v12 - 8);
            v38 = *v56;
            if ( v36 == **v56 )
            {
LABEL_95:
              v11[2] = (__int64)v38;
              goto LABEL_17;
            }
            v86 = 257;
            v39 = v72 == 38;
            v37 = *(__int64 *****)(v12 - 8);
          }
          else
          {
            v37 = (__int64 ****)(v12 - 24LL * (*(_DWORD *)(v12 + 20) & 0xFFFFFFF));
            v38 = *v37;
            if ( v36 == **v37 )
              goto LABEL_95;
            v86 = 257;
            v39 = v72 == 38;
          }
          v23 = (__int64)*v37;
          if ( v36 != **v37 )
          {
            if ( *(_BYTE *)(v23 + 16) > 0x10u )
            {
              v57 = *v37;
              v88 = 257;
              v58 = sub_15FE0A0(v57, (__int64)v36, v39, (__int64)v87, 0);
              if ( v90 )
              {
                v59 = v91;
                v79 = v58;
                sub_157E9D0(v90 + 40, v58);
                v58 = v79;
                v60 = *v59;
                v61 = *(_QWORD *)(v79 + 24);
                *(_QWORD *)(v79 + 32) = v59;
                v60 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v79 + 24) = v60 | v61 & 7;
                *(_QWORD *)(v60 + 8) = v79 + 24;
                *v59 = *v59 & 7 | (v79 + 24);
              }
              v80 = v58;
              sub_164B780(v58, v85);
              v23 = v80;
              if ( v89 )
              {
                v84 = v89;
                sub_1623A60((__int64)&v84, (__int64)v89, 2);
                v23 = v80;
                v62 = *(_QWORD *)(v80 + 48);
                v63 = v80 + 48;
                if ( v62 )
                {
                  sub_161E7C0(v80 + 48, v62);
                  v23 = v80;
                }
                v64 = v84;
                *(_QWORD *)(v23 + 48) = v84;
                if ( v64 )
                {
                  v81 = v23;
                  sub_1623210((__int64)&v84, v64, v63);
                  v23 = v81;
                }
              }
            }
            else
            {
              v23 = sub_15A4750(*v37, v36, v39);
            }
          }
          v40 = *(__int64 **)(a1 + 24);
          v41 = *(unsigned int *)(a1 + 32);
          v42 = &v40[v41];
          v43 = (8 * v41) >> 3;
          if ( (8 * v41) >> 5 )
          {
            v44 = &v40[4 * ((8 * v41) >> 5)];
            while ( v12 != *v40 )
            {
              if ( v12 == v40[1] )
              {
                ++v40;
                goto LABEL_55;
              }
              if ( v12 == v40[2] )
              {
                v40 += 2;
                goto LABEL_55;
              }
              if ( v12 == v40[3] )
              {
                v40 += 3;
                goto LABEL_55;
              }
              v40 += 4;
              if ( v44 == v40 )
              {
                v43 = v42 - v40;
                goto LABEL_80;
              }
            }
            goto LABEL_55;
          }
LABEL_80:
          switch ( v43 )
          {
            case 2LL:
              goto LABEL_109;
            case 3LL:
              if ( v12 == *v40 )
                goto LABEL_55;
              ++v40;
LABEL_109:
              if ( v12 == *v40 )
                goto LABEL_55;
              ++v40;
              break;
            case 1LL:
              break;
            default:
              goto LABEL_84;
          }
          if ( v12 != *v40 )
            goto LABEL_84;
LABEL_55:
          if ( v42 != v40 )
          {
            if ( *(_BYTE *)(v23 + 16) == 60 )
            {
              *v40 = v23;
            }
            else
            {
              if ( v42 != v40 + 1 )
              {
                v73 = v23;
                memmove(v40, v40 + 1, (char *)v42 - (char *)(v40 + 1));
                LODWORD(v41) = *(_DWORD *)(a1 + 32);
                v23 = v73;
              }
              *(_DWORD *)(a1 + 32) = v41 - 1;
            }
            goto LABEL_15;
          }
LABEL_84:
          if ( *(_BYTE *)(v23 + 16) == 60 )
          {
            if ( (unsigned int)v41 >= *(_DWORD *)(a1 + 36) )
            {
              v82 = v23;
              sub_16CD150(a1 + 24, (const void *)(a1 + 40), 0, 8, v14, v15);
              v23 = v82;
              v42 = (__int64 *)(*(_QWORD *)(a1 + 24) + 8LL * *(unsigned int *)(a1 + 32));
            }
            *v42 = v23;
            ++*(_DWORD *)(a1 + 32);
          }
LABEL_15:
          v11[2] = v23;
          if ( *(_BYTE *)(v23 + 16) > 0x17u )
            sub_164B7C0(v23, v12);
LABEL_17:
          if ( v89 )
            sub_161E7C0((__int64)&v89, (__int64)v89);
          v11 += 3;
          if ( v71 == v11 )
            break;
          continue;
        default:
          ++*(_DWORD *)(a1 + 40);
          BUG();
      }
      break;
    }
  }
  v24 = (_QWORD *)sub_1702130(a1, *(_QWORD *)(*(_QWORD *)(a1 + 72) - 24LL), a2);
  v27 = *(_QWORD *)(a1 + 72);
  v28 = (__int64)v24;
  v29 = *(__int64 ***)v27;
  if ( *v24 != *(_QWORD *)v27 )
  {
    v30 = sub_16498A0(*(_QWORD *)(a1 + 72));
    v89 = 0;
    v92 = v30;
    v93 = 0;
    v94 = 0;
    v95 = 0;
    v96 = 0;
    v90 = *(_QWORD *)(v27 + 40);
    v91 = (__int64 *)(v27 + 24);
    v31 = *(unsigned __int8 **)(v27 + 48);
    v87[0] = v31;
    if ( v31 )
    {
      sub_1623A60((__int64)v87, (__int64)v31, 2);
      if ( v89 )
        sub_161E7C0((__int64)&v89, (__int64)v89);
      v89 = v87[0];
      if ( v87[0] )
        sub_1623210((__int64)v87, v87[0], (__int64)&v89);
    }
    v86 = 257;
    if ( v29 != *(__int64 ***)v28 )
    {
      if ( *(_BYTE *)(v28 + 16) > 0x10u )
      {
        v88 = 257;
        v28 = sub_15FE0A0((_QWORD *)v28, (__int64)v29, 0, (__int64)v87, 0);
        if ( v90 )
        {
          v65 = v91;
          sub_157E9D0(v90 + 40, v28);
          v66 = *(_QWORD *)(v28 + 24);
          v67 = *v65;
          *(_QWORD *)(v28 + 32) = v65;
          v67 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v28 + 24) = v67 | v66 & 7;
          *(_QWORD *)(v67 + 8) = v28 + 24;
          *v65 = *v65 & 7 | (v28 + 24);
        }
        sub_164B780(v28, v85);
        if ( v89 )
        {
          v84 = v89;
          sub_1623A60((__int64)&v84, (__int64)v89, 2);
          v68 = *(_QWORD *)(v28 + 48);
          if ( v68 )
            sub_161E7C0(v28 + 48, v68);
          v69 = v84;
          *(_QWORD *)(v28 + 48) = v84;
          if ( v69 )
            sub_1623210((__int64)&v84, v69, v28 + 48);
        }
      }
      else
      {
        v28 = sub_15A4750((__int64 ***)v28, v29, 0);
      }
    }
    if ( *(_BYTE *)(v28 + 16) > 0x17u )
      sub_164B7C0(v28, *(_QWORD *)(a1 + 72));
    if ( v89 )
      sub_161E7C0((__int64)&v89, (__int64)v89);
    v27 = *(_QWORD *)(a1 + 72);
  }
  sub_164D160(v27, v28, a3, a4, a5, a6, v25, v26, a9, a10);
  result = sub_15F20C0(*(_QWORD **)(a1 + 72));
  v33 = *(_QWORD *)(a1 + 120);
  for ( i = *(_QWORD *)(a1 + 112); i != v33; v33 -= 24 )
  {
    while ( 1 )
    {
      v35 = *(_QWORD **)(v33 - 24);
      if ( !v35[1] )
        break;
      v33 -= 24;
      if ( i == v33 )
        return result;
    }
    result = sub_15F20C0(v35);
  }
  return result;
}
