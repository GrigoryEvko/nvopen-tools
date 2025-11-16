// Function: sub_25DFDC0
// Address: 0x25dfdc0
//
__int64 __fastcall sub_25DFDC0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  __int64 *v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 *v7; // r14
  __int64 v8; // r9
  __int64 *v9; // r15
  __int64 v10; // rsi
  __int64 *v11; // rax
  _BYTE *v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // r14
  char v18; // di
  __int64 *v19; // r15
  __int64 v20; // rsi
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // r15
  __int64 v24; // rsi
  __int64 *v25; // r13
  _QWORD *v26; // r14
  __int64 v27; // rbx
  _QWORD *v28; // r13
  char v29; // al
  _QWORD *v30; // rdx
  _QWORD *v31; // rax
  __int64 *v32; // rax
  unsigned __int8 *v33; // rax
  unsigned __int8 v34; // cl
  __int64 v36; // rcx
  char v37; // al
  unsigned __int8 *v38; // r10
  __int64 v39; // rax
  unsigned __int64 *v40; // rcx
  unsigned __int64 v41; // rdx
  __int64 *v42; // rax
  char v43; // dl
  unsigned __int8 *v44; // rax
  bool v45; // al
  unsigned __int8 *v46; // r10
  int v47; // edx
  char v48; // cl
  unsigned __int8 v49; // al
  unsigned __int8 v50; // al
  unsigned __int8 v51; // dl
  char v52; // al
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int8 *v56; // r10
  __int64 *v57; // rax
  __int64 v58; // rdx
  unsigned __int8 *v59; // rax
  bool v60; // al
  bool v61; // zf
  unsigned __int8 v62; // cl
  unsigned __int8 v63; // dl
  unsigned __int8 v64; // al
  __int64 *v65; // rax
  char v66; // al
  unsigned __int8 *v68; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v69; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v70; // [rsp+8h] [rbp-158h]
  unsigned __int8 *v71; // [rsp+18h] [rbp-148h]
  unsigned __int8 v72; // [rsp+28h] [rbp-138h]
  unsigned __int8 *v73; // [rsp+28h] [rbp-138h]
  unsigned __int8 *v74; // [rsp+28h] [rbp-138h]
  __int64 v75; // [rsp+28h] [rbp-138h]
  __int64 *v76; // [rsp+30h] [rbp-130h] BYREF
  __int64 v77; // [rsp+38h] [rbp-128h]
  _BYTE v78[32]; // [rsp+40h] [rbp-120h] BYREF
  __int64 v79; // [rsp+60h] [rbp-100h] BYREF
  __int64 *v80; // [rsp+68h] [rbp-F8h]
  __int64 v81; // [rsp+70h] [rbp-F0h]
  int v82; // [rsp+78h] [rbp-E8h]
  unsigned __int8 v83; // [rsp+7Ch] [rbp-E4h]
  _BYTE v84[32]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v86; // [rsp+A8h] [rbp-B8h]
  __int64 v87; // [rsp+B0h] [rbp-B0h]
  int v88; // [rsp+B8h] [rbp-A8h]
  char v89; // [rsp+BCh] [rbp-A4h]
  _BYTE v90[32]; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v91; // [rsp+E0h] [rbp-80h] BYREF
  _BYTE *v92; // [rsp+E8h] [rbp-78h]
  __int64 v93; // [rsp+F0h] [rbp-70h]
  int v94; // [rsp+F8h] [rbp-68h]
  char v95; // [rsp+FCh] [rbp-64h]
  _BYTE v96[32]; // [rsp+100h] [rbp-60h] BYREF
  _BYTE *v97; // [rsp+120h] [rbp-40h]
  _BYTE *v98; // [rsp+128h] [rbp-38h]

  v86 = (__int64 *)v90;
  v92 = v96;
  v76 = (__int64 *)v78;
  v85 = 0;
  v87 = 4;
  v88 = 0;
  v89 = 1;
  v91 = 0;
  v93 = 4;
  v94 = 0;
  v95 = 1;
  v77 = 0x400000000LL;
  v3 = sub_BAA9B0(a1, (__int64)&v76, 0);
  v7 = v76;
  v79 = 0;
  v8 = 1;
  v97 = v3;
  v80 = (__int64 *)v84;
  v9 = &v76[(unsigned int)v77];
  v83 = 1;
  v81 = 4;
  v82 = 0;
  if ( v76 != v9 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v10 = *v7;
        if ( (_BYTE)v8 )
          break;
LABEL_67:
        ++v7;
        sub_C8CC70((__int64)&v79, v10, (__int64)v4, v5, v6, v8);
        v8 = v83;
        if ( v9 == v7 )
          goto LABEL_8;
      }
      v11 = v80;
      v5 = HIDWORD(v81);
      v4 = &v80[HIDWORD(v81)];
      if ( v80 == v4 )
      {
LABEL_69:
        if ( HIDWORD(v81) >= (unsigned int)v81 )
          goto LABEL_67;
        v5 = (unsigned int)(HIDWORD(v81) + 1);
        ++v7;
        ++HIDWORD(v81);
        *v4 = v10;
        v8 = v83;
        ++v79;
        if ( v9 == v7 )
          break;
      }
      else
      {
        while ( v10 != *v11 )
        {
          if ( v4 == ++v11 )
            goto LABEL_69;
        }
        if ( v9 == ++v7 )
          break;
      }
    }
  }
LABEL_8:
  sub_C8CF80((__int64)&v85, v90, 4, (__int64)v84, (__int64)&v79);
  if ( !v83 )
    _libc_free((unsigned __int64)v80);
  LODWORD(v77) = 0;
  v12 = sub_BAA9B0(a1, (__int64)&v76, 1);
  v17 = v76;
  v18 = 1;
  v79 = 0;
  v98 = v12;
  v80 = (__int64 *)v84;
  v19 = &v76[(unsigned int)v77];
  v83 = 1;
  v81 = 4;
  v82 = 0;
  if ( v76 != v19 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v20 = *v17;
        if ( v18 )
          break;
LABEL_65:
        ++v17;
        sub_C8CC70((__int64)&v79, v20, (__int64)v13, v14, v15, v16);
        v18 = v83;
        if ( v19 == v17 )
          goto LABEL_17;
      }
      v21 = v80;
      v14 = HIDWORD(v81);
      v13 = &v80[HIDWORD(v81)];
      if ( v80 == v13 )
      {
LABEL_72:
        if ( HIDWORD(v81) >= (unsigned int)v81 )
          goto LABEL_65;
        v14 = (unsigned int)(HIDWORD(v81) + 1);
        ++v17;
        ++HIDWORD(v81);
        *v13 = v20;
        v18 = v83;
        ++v79;
        if ( v19 == v17 )
          break;
      }
      else
      {
        while ( v20 != *v21 )
        {
          if ( v13 == ++v21 )
            goto LABEL_72;
        }
        if ( v19 == ++v17 )
          break;
      }
    }
  }
LABEL_17:
  sub_C8CF80((__int64)&v91, v96, 4, (__int64)v84, (__int64)&v79);
  if ( !v83 )
    _libc_free((unsigned __int64)v80);
  if ( v76 != (__int64 *)v78 )
    _libc_free((unsigned __int64)v76);
  v22 = v86;
  if ( v89 )
    v23 = &v86[HIDWORD(v87)];
  else
    v23 = &v86[(unsigned int)v87];
  if ( v86 != v23 )
  {
    while ( 1 )
    {
      v24 = *v22;
      v25 = v22;
      if ( (unsigned __int64)*v22 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v23 == ++v22 )
        goto LABEL_26;
    }
    while ( v25 != v23 )
    {
      if ( v95 )
      {
        v30 = &v92[8 * HIDWORD(v93)];
        v31 = v92;
        if ( v92 != (_BYTE *)v30 )
        {
          while ( *v31 != v24 )
          {
            if ( v30 == ++v31 )
              goto LABEL_38;
          }
          --HIDWORD(v93);
          *v31 = *(_QWORD *)&v92[8 * HIDWORD(v93)];
          ++v91;
        }
      }
      else
      {
        v42 = sub_C8CA60((__int64)&v91, v24);
        if ( v42 )
        {
          *v42 = -2;
          ++v94;
          ++v91;
        }
      }
LABEL_38:
      v32 = v25 + 1;
      if ( v25 + 1 == v23 )
        break;
      while ( 1 )
      {
        v24 = *v32;
        v25 = v32;
        if ( (unsigned __int64)*v32 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v23 == ++v32 )
          goto LABEL_26;
      }
    }
  }
LABEL_26:
  v72 = 0;
  if ( a1 + 40 != *(_QWORD *)(a1 + 48) )
  {
    v26 = *(_QWORD **)(a1 + 48);
    v27 = a1 + 40;
    while ( 1 )
    {
      v28 = v26;
      v26 = (_QWORD *)v26[1];
      if ( (*((_BYTE *)v28 - 41) & 0x10) == 0 && !sub_B2FC80((__int64)(v28 - 6)) && (*(_BYTE *)(v28 - 2) & 0xFu) - 7 > 1 )
      {
        v43 = *((_BYTE *)v28 - 15) & 0xFC;
        *((_BYTE *)v28 - 16) = *(_BYTE *)(v28 - 2) & 0xC0 | 7;
        *((_BYTE *)v28 - 15) = v43 | 0x40;
      }
      if ( (unsigned __int8)sub_25DD4B0((__int64)(v28 - 6), a2, 0, 0) )
        break;
      v29 = *(_BYTE *)(v28 - 2) & 0xF;
      switch ( v29 )
      {
        case 0:
        case 1:
        case 3:
        case 5:
        case 6:
        case 7:
        case 8:
          if ( (*((_BYTE *)v28 - 15) & 0x40) != 0 || ((v29 + 9) & 0xFu) <= 1 || (*(_BYTE *)(v28 - 2) & 0x30) != 0 )
          {
            v71 = (unsigned __int8 *)*(v28 - 10);
            v33 = sub_BD3990(v71, a2);
            if ( *v33 > 3u )
            {
LABEL_47:
              if ( (_QWORD *)v27 == v26 )
                goto LABEL_48;
            }
            else
            {
              v34 = v33[32];
              switch ( v34 & 0xF )
              {
                case 0:
                case 1:
                case 3:
                case 5:
                case 6:
                case 7:
                case 8:
                  if ( (v33[33] & 0x40) == 0 && (((v34 & 0xF) + 9) & 0xFu) > 1 && (v34 & 0x30) == 0 )
                    goto LABEL_47;
                  v68 = v33;
                  sub_AD0030((__int64)v33);
                  v37 = *(_BYTE *)(v28 - 2) & 0xF;
                  if ( ((v37 + 14) & 0xFu) <= 3 )
                    goto LABEL_47;
                  v38 = v68;
                  if ( ((v37 + 7) & 0xFu) <= 1 )
                    goto LABEL_47;
                  v39 = *(v28 - 4);
                  if ( !v39 )
                    goto LABEL_102;
                  if ( *(_QWORD *)(v39 + 8) )
                    goto LABEL_61;
                  if ( v89 )
                  {
                    v57 = v86;
                    v58 = (__int64)&v86[HIDWORD(v87)];
                    if ( v86 != (__int64 *)v58 )
                    {
                      while ( (_QWORD *)*v57 != v28 - 6 )
                      {
                        if ( (__int64 *)v58 == ++v57 )
                          goto LABEL_116;
                      }
LABEL_102:
                      v69 = v38;
                      if ( !sub_25DD3F0((__int64)(v28 - 6), (__int64)&v85) )
                        goto LABEL_47;
                      v59 = sub_BD3990((unsigned __int8 *)*(v28 - 10), (__int64)&v85);
                      v60 = sub_25DD3F0((__int64)v59, (__int64)&v85);
                      v46 = v69;
                      if ( v60 )
                        goto LABEL_47;
LABEL_89:
                      v74 = v46;
                      sub_BD84D0((__int64)(v28 - 6), (__int64)v71);
                      sub_BD6B90(v74, (unsigned __int8 *)v28 - 48);
                      v47 = *(_BYTE *)(v28 - 2) & 0xF;
                      v48 = *(_BYTE *)(v28 - 2) & 0xF;
                      if ( (unsigned int)(v47 - 7) > 1 )
                      {
                        v74[32] = v48 | v74[32] & 0xF0;
                      }
                      else
                      {
                        *((_WORD *)v74 + 16) = *(_BYTE *)(v28 - 2) & 0xF | *((_WORD *)v74 + 16) & 0xFCC0;
                        if ( v47 == 7 )
                          goto LABEL_91;
                      }
                      if ( v47 == 8 )
                      {
LABEL_91:
                        v49 = v74[33] | 0x40;
                        v74[33] = v49;
                        v50 = *((_BYTE *)v28 - 15) & 0x40 | v49 & 0xBF;
                        v51 = v74[32];
                        v74[33] = v50;
                        v74[32] = *(_BYTE *)(v28 - 2) & 0x30 | v51 & 0xCF;
                        goto LABEL_92;
                      }
                      v61 = v48 == 9;
                      v62 = v74[32];
                      if ( (v62 & 0x30) == 0 || v61 )
                      {
                        v50 = *((_BYTE *)v28 - 15) & 0x40 | v74[33] & 0xBF;
                        v63 = v74[32];
                        v74[33] = v50;
                        v74[32] = *(_BYTE *)(v28 - 2) & 0x30 | v63 & 0xCF;
LABEL_110:
                        if ( (v74[32] & 0x30) != 0 && !v61 )
                        {
LABEL_92:
                          v50 |= 0x40u;
                          v74[33] = v50;
                        }
                        v74[33] = *((_BYTE *)v28 - 15) & 3 | v50 & 0xFC;
                        v52 = sub_25DDDB0((__int64)&v85, (__int64)(v28 - 6));
                        v56 = v74;
                        if ( v52 )
                        {
                          sub_25DFCD0((__int64)&v79, (__int64)&v85, (__int64 *)v74, v53, v54, v55);
                          v56 = v74;
                        }
                        v75 = (__int64)v56;
                        if ( (unsigned __int8)sub_25DDDB0((__int64)&v91, (__int64)(v28 - 6)) )
                          sub_AE6EC0((__int64)&v91, v75);
                        goto LABEL_63;
                      }
                      v64 = v74[33] | 0x40;
                      v74[33] = v64;
                      v50 = *((_BYTE *)v28 - 15) & 0x40 | v64 & 0xBF;
                      v74[33] = v50;
                      v74[32] = *(_BYTE *)(v28 - 2) & 0x30 | v62 & 0xCF;
                      if ( v47 != 7 )
                        goto LABEL_110;
                      goto LABEL_92;
                    }
                  }
                  else
                  {
                    v65 = sub_C8CA60((__int64)&v85, (__int64)(v28 - 6));
                    v38 = v68;
                    if ( v65 )
                      goto LABEL_102;
                  }
LABEL_116:
                  v70 = v38;
                  v66 = sub_B19060((__int64)&v91, (__int64)(v28 - 6), v58, v36);
                  v38 = v70;
                  if ( v66 )
                    goto LABEL_102;
LABEL_61:
                  v73 = v38;
                  if ( sub_25DD3F0((__int64)(v28 - 6), (__int64)&v85) )
                  {
                    v44 = sub_BD3990((unsigned __int8 *)*(v28 - 10), (__int64)&v85);
                    v45 = sub_25DD3F0((__int64)v44, (__int64)&v85);
                    v46 = v73;
                    if ( !v45 )
                      goto LABEL_89;
                  }
                  sub_BD84D0((__int64)(v28 - 6), (__int64)v71);
                  if ( sub_25DD3F0((__int64)(v28 - 6), (__int64)&v85) )
                    goto LABEL_46;
LABEL_63:
                  sub_BA8670(v27, (__int64)(v28 - 6));
                  v40 = (unsigned __int64 *)v28[1];
                  v41 = *v28 & 0xFFFFFFFFFFFFFFF8LL;
                  *v40 = v41 | *v40 & 7;
                  *(_QWORD *)(v41 + 8) = v40;
                  *v28 &= 7uLL;
                  v28[1] = 0;
                  sub_AD0030((__int64)(v28 - 6));
                  sub_BD7260((__int64)(v28 - 6), (__int64)(v28 - 6));
                  sub_BD2DD0((__int64)(v28 - 6));
                  v72 = 1;
                  if ( (_QWORD *)v27 == v26 )
                    goto LABEL_48;
                  break;
                case 2:
                case 4:
                case 9:
                case 0xA:
                  goto LABEL_47;
                default:
                  goto LABEL_118;
              }
            }
          }
          else if ( (_QWORD *)v27 == v26 )
          {
            goto LABEL_48;
          }
          break;
        case 2:
        case 4:
        case 9:
        case 10:
          goto LABEL_47;
        default:
LABEL_118:
          BUG();
      }
    }
LABEL_46:
    v72 = 1;
    goto LABEL_47;
  }
LABEL_48:
  if ( v97 )
    sub_25DD070((__int64)v97, (__int64)&v85);
  if ( v98 )
    sub_25DD070((__int64)v98, (__int64)&v91);
  if ( !v95 )
  {
    _libc_free((unsigned __int64)v92);
    if ( v89 )
      return v72;
LABEL_85:
    _libc_free((unsigned __int64)v86);
    return v72;
  }
  if ( !v89 )
    goto LABEL_85;
  return v72;
}
