// Function: sub_1F908A0
// Address: 0x1f908a0
//
__int64 __fastcall sub_1F908A0(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 result; // rax
  __int64 **v6; // r12
  int v8; // r14d
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // rax
  char v12; // r8
  const void **v13; // r13
  __int64 v14; // rdx
  const void **v15; // r9
  char v16; // r11
  char v17; // di
  char v18; // r15
  __int64 v19; // r10
  __int16 v20; // cx
  __int64 v21; // rax
  char v22; // si
  const void **v23; // rax
  int v24; // eax
  char v25; // r9
  char v26; // r9
  int v27; // eax
  char v28; // r9
  int v29; // r13d
  char v30; // r14
  char v31; // r9
  unsigned int v32; // esi
  __int64 *v33; // rdi
  _QWORD *v34; // rax
  int v35; // r10d
  _QWORD *v36; // r8
  unsigned int v37; // edx
  unsigned int v38; // r15d
  __int64 v39; // r11
  int v40; // r15d
  unsigned int v41; // r15d
  _BYTE *v42; // rax
  __int64 v43; // rdx
  _BYTE *i; // rdx
  __int64 v45; // rax
  __int64 v46; // r11
  __int64 v47; // rax
  __int64 **v48; // r15
  __int64 v49; // rbx
  char v50; // r12
  __int64 v51; // r11
  int v52; // r14d
  __int64 v53; // r9
  __int64 v54; // rax
  _QWORD *v55; // rcx
  int v56; // edx
  unsigned int v57; // eax
  _BYTE *v58; // rax
  __int64 v59; // rax
  __int64 *v60; // rdi
  _QWORD *v61; // rax
  char v62; // al
  const void **v63; // rdx
  unsigned int v64; // ebx
  const void **v65; // r14
  _QWORD *v66; // r13
  unsigned __int8 v67; // al
  __int64 *v68; // rsi
  const void **v69; // r8
  __int64 v70; // rdx
  unsigned int v71; // ecx
  __int64 *v72; // r13
  __int64 v73; // rdx
  __int64 v74; // rbx
  unsigned int v75; // eax
  int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 *v79; // rsi
  __int64 v80; // rcx
  const void **v81; // rdx
  __int128 v82; // [rsp-168h] [rbp-168h]
  __int64 v83; // [rsp-150h] [rbp-150h]
  __int64 v84; // [rsp-148h] [rbp-148h]
  unsigned int v85; // [rsp-140h] [rbp-140h]
  __int64 v86; // [rsp-140h] [rbp-140h]
  _QWORD *v87; // [rsp-130h] [rbp-130h]
  __int64 v88; // [rsp-130h] [rbp-130h]
  int v89; // [rsp-130h] [rbp-130h]
  __int64 v90; // [rsp-130h] [rbp-130h]
  _QWORD *v91; // [rsp-130h] [rbp-130h]
  __int64 v92; // [rsp-128h] [rbp-128h]
  _QWORD *v93; // [rsp-128h] [rbp-128h]
  _QWORD *v94; // [rsp-128h] [rbp-128h]
  int v95; // [rsp-128h] [rbp-128h]
  __int64 v96; // [rsp-120h] [rbp-120h]
  char v97; // [rsp-120h] [rbp-120h]
  char v98; // [rsp-120h] [rbp-120h]
  unsigned int v99; // [rsp-120h] [rbp-120h]
  __int64 v100; // [rsp-120h] [rbp-120h]
  int v101; // [rsp-120h] [rbp-120h]
  __int64 v102; // [rsp-118h] [rbp-118h] BYREF
  int v103; // [rsp-110h] [rbp-110h]
  unsigned int v104; // [rsp-108h] [rbp-108h] BYREF
  const void **v105; // [rsp-100h] [rbp-100h]
  unsigned int v106; // [rsp-F8h] [rbp-F8h] BYREF
  const void **v107; // [rsp-F0h] [rbp-F0h]
  char v108; // [rsp-E8h] [rbp-E8h] BYREF
  const void **v109; // [rsp-E0h] [rbp-E0h]
  __int64 v110; // [rsp-D8h] [rbp-D8h] BYREF
  int v111; // [rsp-D0h] [rbp-D0h]
  _BYTE *v112; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v113; // [rsp-C0h] [rbp-C0h]
  _BYTE v114[184]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( (unsigned int)(*(_DWORD *)(a1 + 16) - 1) > 1 )
    return 0;
  v6 = (__int64 **)a1;
  v8 = *(_DWORD *)(a2 + 56);
  v9 = *(_QWORD *)(a2 + 72);
  v102 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v102, v9, 2);
  v10 = *(_DWORD *)(a2 + 64);
  LOBYTE(v106) = 1;
  v107 = 0;
  v103 = v10;
  v11 = *(_QWORD *)(a2 + 40);
  v12 = *(_BYTE *)v11;
  v13 = *(const void ***)(v11 + 8);
  LOBYTE(v104) = *(_BYTE *)v11;
  v105 = v13;
  if ( v8 )
  {
    v14 = *(_QWORD *)(a2 + 32);
    v15 = 0;
    v16 = 0;
    v17 = 1;
    v18 = 1;
    v19 = v14 + 40LL * (unsigned int)(v8 - 1) + 40;
    do
    {
      v20 = *(_WORD *)(*(_QWORD *)v14 + 24LL);
      if ( v20 != 48 )
      {
        if ( (unsigned __int16)(v20 - 143) > 1u )
          goto LABEL_19;
        v21 = *(_QWORD *)(**(_QWORD **)(*(_QWORD *)v14 + 32LL) + 40LL)
            + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)v14 + 32LL) + 8LL);
        v22 = *(_BYTE *)v21;
        v23 = *(const void ***)(v21 + 8);
        if ( v17 == 1 )
        {
          v17 = v22;
          v16 = 1;
        }
        else
        {
          if ( v22 != v17 || v23 != v15 && !v17 )
          {
LABEL_19:
            LOBYTE(v106) = 1;
            v107 = 0;
            goto LABEL_20;
          }
          v23 = v15;
        }
        v15 = v23;
        v18 &= v20 == 144;
      }
      v14 += 40;
    }
    while ( v14 != v19 );
    if ( v16 )
    {
      LOBYTE(v106) = v17;
      v107 = v15;
    }
  }
  else
  {
    v18 = 1;
  }
LABEL_20:
  if ( !v12 )
  {
    if ( sub_1F58D20((__int64)&v104) )
    {
      v62 = sub_1F596B0((__int64)&v104);
      v26 = v106;
      v108 = v62;
      v12 = v62;
      v109 = v63;
      if ( (_BYTE)v106 == 1 )
        goto LABEL_24;
      if ( v62 )
      {
LABEL_23:
        v24 = sub_1F6C8D0(v12);
        if ( !v24 )
          goto LABEL_24;
LABEL_31:
        if ( (v24 & (v24 - 1)) != 0 )
          goto LABEL_24;
        if ( v25 )
        {
          v27 = sub_1F6C8D0(v25);
        }
        else
        {
          v27 = sub_1F58D40((__int64)&v106);
          v28 = 0;
        }
        v98 = v28;
        if ( !v27 )
          goto LABEL_24;
        v29 = v27 & (v27 - 1);
        if ( v29 )
          goto LABEL_24;
        v30 = *(_BYTE *)sub_1E0A0C0((*v6)[4]);
        if ( v108 )
        {
          v99 = sub_1F6C8D0(v108);
        }
        else
        {
          v75 = sub_1F58D40((__int64)&v108);
          v31 = v98;
          v99 = v75;
        }
        if ( v31 )
          v32 = sub_1F6C8D0(v31);
        else
          v32 = sub_1F58D40((__int64)&v106);
        v33 = *v6;
        if ( v18 )
        {
          v112 = 0;
          LODWORD(v113) = 0;
          v34 = sub_1D2B300(v33, 0x30u, (__int64)&v112, v106, (__int64)v107, (__int64)&v112);
          v35 = v99 / v32;
          v36 = v34;
          v38 = v37;
          if ( v112 )
          {
            v87 = v34;
            sub_161E7C0((__int64)&v112, (__int64)v112);
            v36 = v87;
            v35 = v99 / v32;
          }
          v39 = v38;
        }
        else
        {
          v77 = sub_1D38BB0((__int64)v33, 0, (__int64)&v102, v106, v107, 0, a3, a4, a5, 0);
          v35 = v99 / v32;
          v36 = (_QWORD *)v77;
          v39 = v78;
        }
        if ( (_BYTE)v104 )
        {
          v40 = word_42FA680[(unsigned __int8)(v104 - 14)];
        }
        else
        {
          v90 = v39;
          v94 = v36;
          v101 = v35;
          v76 = sub_1F58D30((__int64)&v104);
          v39 = v90;
          v36 = v94;
          v35 = v101;
          v40 = v76;
        }
        v41 = v35 * v40;
        v42 = v114;
        v112 = v114;
        v113 = 0x800000000LL;
        v43 = v41;
        if ( v41 > 8 )
        {
          v86 = v39;
          v91 = v36;
          v95 = v35;
          sub_16CD150((__int64)&v112, v114, v41, 16, (int)v36, (int)&v112);
          v42 = v112;
          v39 = v86;
          v36 = v91;
          v35 = v95;
          v43 = v41;
        }
        LODWORD(v113) = v41;
        for ( i = &v42[16 * v43]; i != v42; v42 += 16 )
        {
          if ( v42 )
          {
            *(_QWORD *)v42 = v36;
            *((_QWORD *)v42 + 1) = v39;
          }
        }
        v45 = *(unsigned int *)(a2 + 56);
        if ( (_DWORD)v45 )
        {
          v46 = 5 * v45;
          v85 = v41;
          v47 = a2;
          v48 = v6;
          v49 = 0;
          v50 = v30;
          v51 = 8 * v46;
          v52 = v35;
          v53 = v47;
          do
          {
            v59 = *(_QWORD *)(*(_QWORD *)(v53 + 32) + v49);
            if ( *(_WORD *)(v59 + 24) == 48 )
            {
              v60 = *v48;
              v88 = v51;
              v92 = v53;
              v110 = 0;
              v111 = 0;
              v61 = sub_1D2B300(v60, 0x30u, (__int64)&v110, v106, (__int64)v107, v53);
              v53 = v92;
              v51 = v88;
              v55 = v61;
              if ( v110 )
              {
                v83 = v88;
                v84 = v92;
                v89 = v56;
                v93 = v61;
                sub_161E7C0((__int64)&v110, v110);
                v51 = v83;
                v53 = v84;
                v56 = v89;
                v55 = v93;
              }
            }
            else
            {
              v54 = *(_QWORD *)(v59 + 32);
              v55 = *(_QWORD **)v54;
              v56 = *(_DWORD *)(v54 + 8);
            }
            v57 = v29 + v52 - 1;
            if ( !v50 )
              v57 = v29;
            v49 += 40;
            v29 += v52;
            v58 = &v112[16 * v57];
            *(_QWORD *)v58 = v55;
            *((_DWORD *)v58 + 2) = v56;
          }
          while ( v51 != v49 );
          v6 = v48;
          v41 = v85;
        }
        v64 = v106;
        v65 = v107;
        v66 = (_QWORD *)(*v6)[6];
        v67 = sub_1D15020(v106, v41);
        if ( v67 )
        {
          if ( !*((_BYTE *)v6 + 25) )
          {
            v68 = v6[1];
            v69 = 0;
            if ( v67 == 1 )
            {
              v70 = 1;
              goto LABEL_89;
            }
LABEL_113:
            v70 = v67;
            v80 = v67;
            goto LABEL_104;
          }
          v79 = v6[1];
          v69 = 0;
        }
        else
        {
          v67 = sub_1F593D0(v66, v64, (__int64)v65, v41);
          v69 = v81;
          if ( !*((_BYTE *)v6 + 25) )
          {
            v68 = v6[1];
            if ( v67 == 1 )
            {
              v70 = 1;
              goto LABEL_89;
            }
            if ( !v67 )
            {
LABEL_90:
              v71 = 1;
              if ( (_BYTE)v104 != 1 )
              {
                if ( !(_BYTE)v104 )
                  goto LABEL_93;
                v71 = (unsigned __int8)v104;
                if ( !v68[(unsigned __int8)v104 + 15] )
                  goto LABEL_93;
              }
              if ( *((_BYTE *)v68 + 259 * v71 + 2526) )
                goto LABEL_93;
LABEL_108:
              result = 0;
LABEL_94:
              if ( v112 != v114 )
              {
                v100 = result;
                _libc_free((unsigned __int64)v112);
                result = v100;
              }
              goto LABEL_25;
            }
            goto LABEL_113;
          }
          if ( !v67 )
            goto LABEL_108;
          v79 = v6[1];
        }
        v80 = v67;
        v70 = v67;
        if ( !v79[v67 + 15] )
          goto LABEL_108;
        v68 = v6[1];
        if ( v67 == 1 )
          goto LABEL_89;
LABEL_104:
        if ( v68[v80 + 15] )
        {
LABEL_89:
          if ( !*((_BYTE *)v68 + 259 * v70 + 2526) )
          {
LABEL_93:
            *((_QWORD *)&v82 + 1) = (unsigned int)v113;
            *(_QWORD *)&v82 = v112;
            v72 = sub_1D359D0(*v6, 104, (__int64)&v102, v67, v69, 0, *(double *)a3.m128i_i64, a4, a5, v82);
            v74 = v73;
            sub_1F81BC0((__int64)v6, (__int64)v72);
            result = sub_1D32840(
                       *v6,
                       v104,
                       v105,
                       (__int64)v72,
                       v74,
                       *(double *)a3.m128i_i64,
                       a4,
                       *(double *)a5.m128i_i64);
            goto LABEL_94;
          }
          goto LABEL_90;
        }
        goto LABEL_90;
      }
    }
    else
    {
      v26 = v106;
      v108 = 0;
      v109 = v13;
      if ( (_BYTE)v106 == 1 )
        goto LABEL_24;
    }
    v97 = v26;
    v24 = sub_1F58D40((__int64)&v108);
    v25 = v97;
    if ( !v24 )
      goto LABEL_24;
    goto LABEL_31;
  }
  if ( (unsigned __int8)(v12 - 14) <= 0x5Fu )
  {
    switch ( v12 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v108 = 3;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v108 = 4;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v108 = 5;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v108 = 6;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 6;
        break;
      case 55:
        v108 = 7;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v108 = 8;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v108 = 9;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v108 = 10;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 10;
        break;
      default:
        v108 = 2;
        v109 = 0;
        if ( (_BYTE)v106 == 1 )
          goto LABEL_24;
        v12 = 2;
        break;
    }
    goto LABEL_23;
  }
  v108 = v12;
  v109 = v13;
  if ( (_BYTE)v106 != 1 )
    goto LABEL_23;
LABEL_24:
  result = 0;
LABEL_25:
  if ( v102 )
  {
    v96 = result;
    sub_161E7C0((__int64)&v102, v102);
    return v96;
  }
  return result;
}
