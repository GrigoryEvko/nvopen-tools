// Function: sub_2AD24E0
// Address: 0x2ad24e0
//
unsigned __int64 __fastcall sub_2AD24E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // r13d
  unsigned int v7; // eax
  _BYTE *v8; // r8
  unsigned __int64 v9; // rcx
  int v10; // ecx
  unsigned int v11; // edx
  unsigned __int8 *v12; // r9
  __int64 v13; // rdx
  __int64 v14; // rdi
  unsigned __int8 *v15; // rsi
  int v16; // edx
  unsigned __int64 v17; // rax
  int v18; // edx
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  char v21; // al
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int64 v27; // rax
  unsigned __int8 **v28; // rbx
  unsigned __int8 *v29; // r14
  __int64 v30; // rax
  signed __int64 v31; // rax
  int v32; // edx
  __int64 *v33; // rax
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 *v39; // r14
  __int64 *v40; // rbx
  __int64 v41; // rax
  unsigned __int64 v42; // rdx
  int v43; // r10d
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // rdi
  __int64 v47; // r11
  __int64 *v48; // r10
  __int64 *v49; // r14
  unsigned __int64 v50; // rbx
  __int64 j; // r10
  __int64 v52; // rax
  int v53; // edx
  bool v54; // zf
  bool v55; // of
  __int64 *v56; // r11
  unsigned __int64 v57; // rax
  _QWORD *v58; // rax
  unsigned __int8 *v59; // rdx
  _QWORD *v60; // rax
  _QWORD *v61; // r11
  __int64 *v62; // r10
  __int64 *v63; // rbx
  unsigned __int64 v64; // r14
  __int64 i; // r10
  __int64 v66; // rax
  int v67; // edx
  int v68; // edx
  __int64 *v69; // r11
  unsigned __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdx
  int v73; // edx
  bool v74; // cc
  unsigned __int64 v75; // rax
  unsigned __int8 **v76; // [rsp+0h] [rbp-150h]
  int v78; // [rsp+28h] [rbp-128h]
  __int64 v79; // [rsp+30h] [rbp-120h]
  __int64 v80; // [rsp+30h] [rbp-120h]
  unsigned __int64 v81; // [rsp+38h] [rbp-118h]
  __int64 v82; // [rsp+48h] [rbp-108h]
  __int64 v83; // [rsp+48h] [rbp-108h]
  unsigned __int8 **v84; // [rsp+50h] [rbp-100h]
  unsigned __int64 v85; // [rsp+58h] [rbp-F8h]
  _QWORD *v86; // [rsp+60h] [rbp-F0h]
  unsigned __int8 *v87; // [rsp+68h] [rbp-E8h]
  __int64 *v88; // [rsp+68h] [rbp-E8h]
  __int64 v89; // [rsp+70h] [rbp-E0h]
  __int64 *v90; // [rsp+70h] [rbp-E0h]
  __int64 *v91; // [rsp+70h] [rbp-E0h]
  unsigned __int8 **v92; // [rsp+78h] [rbp-D8h]
  __int64 *v93; // [rsp+78h] [rbp-D8h]
  int v94; // [rsp+80h] [rbp-D0h]
  unsigned int v95; // [rsp+84h] [rbp-CCh]
  _QWORD *v96; // [rsp+88h] [rbp-C8h]
  unsigned __int8 *v97; // [rsp+98h] [rbp-B8h] BYREF
  _QWORD *v98; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v99; // [rsp+A8h] [rbp-A8h] BYREF
  unsigned __int64 v100; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v101; // [rsp+B8h] [rbp-98h] BYREF
  __int64 v102; // [rsp+C0h] [rbp-90h] BYREF
  unsigned int v103; // [rsp+C8h] [rbp-88h]
  _BYTE *v104; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-78h]
  _BYTE v106[112]; // [rsp+E0h] [rbp-70h] BYREF

  v6 = a4;
  v104 = v106;
  v105 = 0x800000000LL;
  HIDWORD(v96) = HIDWORD(a4);
  sub_9C95B0((__int64)&v104, a2);
  v7 = v105;
  if ( (_DWORD)v105 )
  {
    v8 = v104;
    v94 = 0;
    v85 = 0;
    v9 = 0;
    if ( v6 )
      v9 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v6;
    v86 = (_QWORD *)v9;
    while ( 1 )
    {
      v13 = v7--;
      v14 = *(_QWORD *)(a3 + 8);
      v15 = *(unsigned __int8 **)&v8[8 * v13 - 8];
      v16 = *(_DWORD *)(a3 + 24);
      LODWORD(v105) = v7;
      v97 = v15;
      if ( !v16 )
        goto LABEL_8;
      v10 = v16 - 1;
      v11 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v12 = *(unsigned __int8 **)(v14 + 24LL * v11);
      if ( v15 == v12 )
      {
LABEL_6:
        if ( !v7 )
          goto LABEL_35;
      }
      else
      {
        v43 = 1;
        while ( v12 != (unsigned __int8 *)-4096LL )
        {
          v11 = v10 & (v43 + v11);
          v12 = *(unsigned __int8 **)(v14 + 24LL * v11);
          if ( v15 == v12 )
            goto LABEL_6;
          ++v43;
        }
LABEL_8:
        LODWORD(v96) = v6;
        v17 = sub_2AD0150(a1, v15, (__int64)v96);
        BYTE4(v99) = 0;
        v78 = v18;
        LODWORD(v99) = 1;
        v81 = v17;
        v19 = sub_2AD0150(a1, v97, v99);
        v103 = 0;
        v100 = v19;
        v102 = v6;
        v101 = v20;
        sub_2AA9150((__int64)&v102, (__int64)&v100);
        v89 = v102;
        v95 = v103;
        v21 = sub_2AC3650(a1, v97, (unsigned __int64)v96);
        v25 = (__int64)v97;
        if ( !v21 )
          goto LABEL_10;
        v26 = *((_QWORD *)v97 + 1);
        if ( *(_BYTE *)(v26 + 8) == 7 )
          goto LABEL_10;
        v60 = sub_2AAEED0(v26, (__int64)v96, (__int64)v97, v22, v23, v24);
        v54 = *((_BYTE *)v60 + 8) == 15;
        v98 = v60;
        v61 = v60;
        if ( !v54 )
        {
          v62 = (__int64 *)&v98;
          v93 = &v99;
LABEL_91:
          v80 = a3;
          v63 = v62 + 1;
          v64 = v89;
          for ( i = (__int64)v61; ; i = *v63++ )
          {
            v103 = v6;
            v69 = *(__int64 **)(a1 + 448);
            if ( v6 <= 0x40 )
            {
              v102 = (__int64)v86;
            }
            else
            {
              v83 = i;
              v91 = *(__int64 **)(a1 + 448);
              sub_C43690((__int64)&v102, -1, 1);
              i = v83;
              v69 = v91;
            }
            v66 = sub_DFAAD0(v69, i, (__int64)&v102, 1u, 0);
            v54 = v67 == 1;
            v68 = 1;
            if ( !v54 )
              v68 = v95;
            v55 = __OFADD__(v66, v64);
            v64 += v66;
            v95 = v68;
            if ( v55 )
            {
              v64 = 0x8000000000000000LL;
              if ( v66 > 0 )
                v64 = 0x7FFFFFFFFFFFFFFFLL;
            }
            if ( v103 > 0x40 && v102 )
              j_j___libc_free_0_0(v102);
            if ( v63 == v93 )
              break;
          }
          v89 = v64;
          a3 = v80;
          goto LABEL_114;
        }
        v62 = (__int64 *)v60[2];
        v93 = &v62[*((unsigned int *)v60 + 3)];
        if ( v93 != v62 )
        {
          v61 = (_QWORD *)*v62;
          goto LABEL_91;
        }
LABEL_114:
        v71 = sub_DFD270(*(_QWORD *)(a1 + 448), 55, *(_DWORD *)(a1 + 992));
        v103 = 0;
        v100 = v71;
        v101 = v72;
        v102 = v6;
        sub_2AA9150((__int64)&v102, (__int64)&v100);
        v73 = 1;
        if ( v103 != 1 )
          v73 = v95;
        v95 = v73;
        if ( __OFADD__(v102, v89) )
        {
          v22 = 0x7FFFFFFFFFFFFFFFLL;
          v75 = 0x8000000000000000LL;
          if ( v102 > 0 )
            v75 = 0x7FFFFFFFFFFFFFFFLL;
          v89 = v75;
        }
        else
        {
          v89 += v102;
        }
        v25 = (__int64)v97;
LABEL_10:
        v27 = 4LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v25 + 7) & 0x40) != 0 )
        {
          v28 = *(unsigned __int8 ***)(v25 - 8);
          v92 = &v28[v27];
        }
        else
        {
          v92 = (unsigned __int8 **)v25;
          v28 = (unsigned __int8 **)(v25 - v27 * 8);
        }
        if ( v28 != v92 )
        {
          v79 = a3;
          while ( 1 )
          {
            while ( 1 )
            {
              v29 = *v28;
              if ( **v28 <= 0x1Cu )
                goto LABEL_21;
              v30 = *((_QWORD *)v29 + 2);
              if ( !v30 || *(_QWORD *)(v30 + 8) || (v22 = *((_QWORD *)v29 + 5), *(_QWORD *)(a2 + 40) != v22) )
              {
                LODWORD(v96) = v6;
                v102 = (__int64)v96;
LABEL_17:
                if ( !BYTE4(v102) && v6 == 1
                  || !(unsigned __int8)sub_B19060(*(_QWORD *)(a1 + 416) + 56LL, *((_QWORD *)v29 + 5), v25, v22)
                  || (unsigned __int8)sub_D48480(*(_QWORD *)(a1 + 416), (__int64)v29, v25, v22)
                  || (unsigned int)sub_2AAA2B0(a1, (__int64)v29, v6, SBYTE4(v102)) == 5
                  || sub_2ABFD00(a1 + 192, (__int64)&v102) && (unsigned __int8)sub_2AB2DA0(a1, (__int64)v29, v102) )
                {
                  goto LABEL_21;
                }
                v46 = *((_QWORD *)v29 + 1);
                v102 = (__int64)v96;
                if ( *(_BYTE *)(v46 + 8) == 15 )
                  v47 = (__int64)sub_E454C0(v46, (__int64)v96, v25, v22, v44, v45);
                else
                  v47 = sub_2AAEDF0(v46, (__int64)v96);
                v100 = v47;
                if ( *(_BYTE *)(v47 + 8) == 15 )
                {
                  v48 = *(__int64 **)(v47 + 16);
                  v88 = &v48[*(unsigned int *)(v47 + 12)];
                  if ( v88 != v48 )
                  {
                    v47 = *v48;
                    goto LABEL_65;
                  }
LABEL_21:
                  v28 += 4;
                  if ( v92 == v28 )
                    goto LABEL_22;
                }
                else
                {
                  v48 = (__int64 *)&v100;
                  v88 = &v101;
LABEL_65:
                  v76 = v28;
                  v49 = v48 + 1;
                  v50 = v89;
                  for ( j = v47; ; j = *v49++ )
                  {
                    v103 = v6;
                    v56 = *(__int64 **)(a1 + 448);
                    if ( v6 <= 0x40 )
                    {
                      v102 = (__int64)v86;
                    }
                    else
                    {
                      v82 = j;
                      v90 = *(__int64 **)(a1 + 448);
                      sub_C43690((__int64)&v102, -1, 1);
                      j = v82;
                      v56 = v90;
                    }
                    v52 = sub_DFAAD0(v56, j, (__int64)&v102, 0, 1u);
                    v22 = 0;
                    v54 = v53 == 1;
                    v25 = 1;
                    if ( !v54 )
                      v25 = v95;
                    v55 = __OFADD__(v52, v50);
                    v50 += v52;
                    v95 = v25;
                    if ( v55 )
                    {
                      v50 = 0x8000000000000000LL;
                      if ( v52 > 0 )
                        v50 = 0x7FFFFFFFFFFFFFFFLL;
                    }
                    if ( v103 > 0x40 && v102 )
                      j_j___libc_free_0_0(v102);
                    if ( v88 == v49 )
                      break;
                  }
                  v89 = v50;
                  v28 = v76 + 4;
                  if ( v92 == v76 + 4 )
                  {
LABEL_22:
                    a3 = v79;
                    goto LABEL_23;
                  }
                }
                continue;
              }
              LODWORD(v96) = v6;
              if ( !(unsigned __int8)sub_2AB2DA0(a1, (__int64)v29, (unsigned __int64)v96)
                && !sub_2AC3650(a1, v29, (unsigned __int64)v96) )
              {
                break;
              }
LABEL_110:
              LODWORD(v96) = v6;
              v25 = *v29;
              v102 = (__int64)v96;
              if ( (unsigned __int8)v25 > 0x1Cu )
                goto LABEL_17;
              v28 += 4;
              if ( v92 == v28 )
                goto LABEL_22;
            }
            v35 = sub_1168D40((__int64)v29);
            if ( (__int64 *)v35 != v36 )
            {
              v87 = v29;
              v39 = v36;
              v84 = v28;
              v40 = (__int64 *)v35;
              while ( 1 )
              {
                if ( *(_BYTE *)*v40 > 0x1Cu )
                {
                  LODWORD(v96) = v6;
                  if ( (unsigned __int8)sub_2AB2C60(a1, *v40, (__int64)v96) )
                    break;
                }
                v40 += 4;
                if ( v39 == v40 )
                {
                  v29 = v87;
                  v28 = v84;
                  goto LABEL_48;
                }
              }
              v29 = v87;
              v28 = v84;
              goto LABEL_110;
            }
LABEL_48:
            v41 = (unsigned int)v105;
            v22 = HIDWORD(v105);
            v42 = (unsigned int)v105 + 1LL;
            if ( v42 > HIDWORD(v105) )
            {
              sub_C8D5F0((__int64)&v104, v106, v42, 8u, v37, v38);
              v41 = (unsigned int)v105;
            }
            v25 = (__int64)v104;
            v28 += 4;
            *(_QWORD *)&v104[8 * v41] = v29;
            LODWORD(v105) = v105 + 1;
            if ( v92 == v28 )
              goto LABEL_22;
          }
        }
LABEL_23:
        if ( *(_DWORD *)(a1 + 992) != 2 )
          v89 /= 2;
        if ( v95 == 1 )
        {
          v31 = v81 - v89;
          if ( __OFSUB__(v81, v89) )
          {
            if ( v89 <= 0 )
            {
LABEL_120:
              v70 = v85 + 0x7FFFFFFFFFFFFFFFLL;
              if ( __OFADD__(v85, 0x7FFFFFFFFFFFFFFFLL) )
              {
                v94 = 1;
LABEL_122:
                v85 = 0x7FFFFFFFFFFFFFFFLL;
                goto LABEL_32;
              }
            }
            else
            {
LABEL_104:
              v70 = v85 + 0x8000000000000000LL;
              if ( __OFADD__(v85, 0x8000000000000000LL) )
              {
                v94 = 1;
                v85 = 0x8000000000000000LL;
                goto LABEL_32;
              }
            }
            v85 = v70;
            v94 = 1;
            goto LABEL_32;
          }
          v94 = 1;
        }
        else
        {
          v31 = v81 - v89;
          if ( __OFSUB__(v81, v89) )
          {
            if ( v89 <= 0 )
            {
              if ( v78 == 1 )
                goto LABEL_120;
              v57 = v85 + 0x7FFFFFFFFFFFFFFFLL;
              if ( __OFADD__(v85, 0x7FFFFFFFFFFFFFFFLL) )
                goto LABEL_122;
            }
            else
            {
              if ( v78 == 1 )
                goto LABEL_104;
              v57 = v85 + 0x8000000000000000LL;
              if ( __OFADD__(v85, 0x8000000000000000LL) )
              {
                v85 = 0x8000000000000000LL;
                goto LABEL_32;
              }
            }
            goto LABEL_80;
          }
          v32 = 1;
          if ( v78 != 1 )
            v32 = v94;
          v94 = v32;
        }
        if ( __OFADD__(v31, v85) )
        {
          v74 = v31 <= 0;
          v57 = 0x8000000000000000LL;
          if ( !v74 )
            v57 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_80:
          v85 = v57;
          goto LABEL_32;
        }
        v85 += v31;
LABEL_32:
        if ( (unsigned __int8)sub_2AC1850(a3, (__int64 *)&v97, &v102) )
        {
          v33 = (__int64 *)(v102 + 8);
        }
        else
        {
          v58 = sub_2AD00C0(a3, (__int64 *)&v97, (_QWORD *)v102);
          v59 = v97;
          v58[1] = 0;
          v33 = v58 + 1;
          *(v33 - 1) = (__int64)v59;
          *((_DWORD *)v33 + 2) = 0;
        }
        *v33 = v89;
        v8 = v104;
        *((_DWORD *)v33 + 2) = v95;
        v7 = v105;
        if ( !(_DWORD)v105 )
          goto LABEL_35;
      }
    }
  }
  v8 = v104;
  v85 = 0;
LABEL_35:
  if ( v8 != v106 )
    _libc_free((unsigned __int64)v8);
  return v85;
}
