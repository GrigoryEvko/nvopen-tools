// Function: sub_259BD00
// Address: 0x259bd00
//
__int64 __fastcall sub_259BD00(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  int v6; // ecx
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __m128i v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdx
  __int32 v15; // r15d
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // rax
  int v20; // r15d
  unsigned __int64 v21; // rdx
  _QWORD *v22; // rax
  int v23; // eax
  char v24; // al
  __int64 v25; // r15
  __int64 v26; // rax
  int v27; // esi
  __int64 v28; // rdi
  int v29; // esi
  unsigned int v30; // ecx
  __int64 *v31; // rdx
  __int64 v32; // r9
  int v34; // ecx
  __int64 v35; // r11
  __int64 v36; // rdi
  __int64 v37; // rax
  unsigned int v38; // esi
  __int64 *v39; // rdx
  __int64 v40; // r8
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r11
  unsigned __int64 v47; // rsi
  unsigned __int64 v48; // r9
  unsigned int v49; // eax
  int v50; // edx
  __int64 v51; // r8
  unsigned int v52; // r10d
  __int64 v53; // rdx
  __int64 v54; // r15
  __int64 v55; // rbx
  __int64 v56; // r12
  __int64 v57; // r15
  __int64 v58; // r13
  unsigned __int8 *v59; // rax
  __int64 v60; // r9
  int v61; // edx
  unsigned __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // rdx
  __int64 v65; // rsi
  __int64 v66; // r8
  unsigned int v67; // r10d
  __int64 v68; // rcx
  __int64 v69; // rdi
  __int64 v70; // rdx
  int v71; // ecx
  __int64 v72; // rax
  __int64 v73; // rbx
  int v74; // edx
  int v75; // r8d
  __int64 v76; // rdx
  int v77; // ecx
  unsigned int v78; // ecx
  __int64 v79; // r8
  __int64 v80; // [rsp+0h] [rbp-180h]
  unsigned __int64 v81; // [rsp+8h] [rbp-178h]
  __int64 v82; // [rsp+8h] [rbp-178h]
  __int64 v83; // [rsp+10h] [rbp-170h]
  unsigned int v84; // [rsp+18h] [rbp-168h]
  __int64 v85; // [rsp+18h] [rbp-168h]
  __int64 v86; // [rsp+18h] [rbp-168h]
  __int64 v87; // [rsp+18h] [rbp-168h]
  __int64 v88; // [rsp+20h] [rbp-160h]
  __int64 v89; // [rsp+28h] [rbp-158h]
  __m128i *v90; // [rsp+40h] [rbp-140h]
  char v91; // [rsp+52h] [rbp-12Eh]
  char v92; // [rsp+53h] [rbp-12Dh]
  unsigned int v93; // [rsp+54h] [rbp-12Ch]
  __int64 *v94; // [rsp+58h] [rbp-128h]
  __int64 v95; // [rsp+60h] [rbp-120h]
  __int64 v96; // [rsp+68h] [rbp-118h]
  bool v97; // [rsp+75h] [rbp-10Bh] BYREF
  char v98; // [rsp+76h] [rbp-10Ah] BYREF
  char v99; // [rsp+77h] [rbp-109h] BYREF
  __int64 v100; // [rsp+78h] [rbp-108h] BYREF
  _BYTE *v101; // [rsp+80h] [rbp-100h]
  __int64 v102; // [rsp+88h] [rbp-F8h]
  _QWORD *v103; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v104; // [rsp+98h] [rbp-E8h]
  char v105; // [rsp+A0h] [rbp-E0h]
  char *v106; // [rsp+B0h] [rbp-D0h]
  __int64 v107; // [rsp+B8h] [rbp-C8h]
  __int64 v108; // [rsp+C0h] [rbp-C0h]
  __int64 *v109; // [rsp+C8h] [rbp-B8h]
  bool *v110; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v111; // [rsp+D8h] [rbp-A8h]
  __int64 v112; // [rsp+E0h] [rbp-A0h]
  unsigned int v113; // [rsp+E8h] [rbp-98h]
  __int64 v114; // [rsp+F8h] [rbp-88h]
  __int64 v115; // [rsp+100h] [rbp-80h]
  __int64 v116; // [rsp+108h] [rbp-78h]
  __m128i v117; // [rsp+110h] [rbp-70h] BYREF
  __int64 v118; // [rsp+120h] [rbp-60h]
  __int64 v119; // [rsp+128h] [rbp-58h]
  __int64 v120; // [rsp+138h] [rbp-48h]
  __int64 v121; // [rsp+140h] [rbp-40h]
  __int64 v122; // [rsp+148h] [rbp-38h]

  v3 = a2;
  v90 = (__m128i *)(a1 + 72);
  v89 = sub_25096F0((_QWORD *)(a1 + 72));
  v94 = (__int64 *)sub_2555710(*(_QWORD *)(*(_QWORD *)(a2 + 208) + 240LL), v89, 0);
  sub_250D230((unsigned __int64 *)&v117, v89, 4, 0);
  v100 = sub_251BBC0(a2, v117.m128i_i64[0], v117.m128i_i64[1], a1, 2, 0, 1);
  v4 = *(_QWORD *)(a2 + 208);
  v5 = *(_QWORD *)(v4 + 120);
  v6 = *(_DWORD *)(v4 + 376);
  v7 = *(_QWORD *)(v4 + 240);
  v83 = v5;
  v97 = ((v6 - 26) & 0xFFFFFFEE) != 0;
  v8 = *(_QWORD *)v7;
  if ( *(_QWORD *)v7 )
  {
    if ( !*(_BYTE *)(v7 + 16) )
    {
      v9 = sub_BC1CD0(v8, &unk_4F875F0, v89);
LABEL_4:
      v88 = v9 + 8;
      goto LABEL_5;
    }
    v9 = sub_BBB550(v8, (__int64)&unk_4F875F0, v89);
    if ( v9 )
      goto LABEL_4;
  }
  v88 = 0;
LABEL_5:
  v10 = *(_QWORD *)(a1 + 136);
  v98 = 0;
  v106 = &v98;
  v109 = &v100;
  v11 = *(unsigned int *)(a1 + 144);
  v107 = a1;
  v108 = a2;
  v96 = v10 + 16 * v11;
  if ( v10 != v96 )
  {
    v92 = 0;
    v91 = 0;
    v93 = 1;
    v95 = a1;
    while ( 1 )
    {
      v16 = *(_QWORD *)(v10 + 8);
      if ( *(_DWORD *)(v16 + 12) == 2 )
        goto LABEL_17;
      v17 = sub_D5CD40(*(_QWORD *)v16, v94);
      if ( v17 )
        break;
LABEL_20:
      v117.m128i_i64[0] = v3;
      v117.m128i_i64[1] = v95;
      v18 = sub_25096F0(v90);
      v19 = (__int64 *)sub_2555710(*(_QWORD *)(*(_QWORD *)(v3 + 208) + 240LL), v18, 0);
      sub_D5CDD0(
        (__int64)&v103,
        *(_QWORD *)v16,
        v19,
        (__int64 (__fastcall *)(__int64, _QWORD))sub_254E230,
        (__int64)&v117);
      v20 = *(_DWORD *)(v16 + 8);
      if ( v20 != 109 && (_DWORD)qword_4FEF908 != -1 )
      {
        if ( !v105 )
          goto LABEL_62;
        v21 = (int)qword_4FEF908;
        v84 = v104;
        if ( v104 > 0x40 )
        {
          v81 = (int)qword_4FEF908;
          if ( v84 - (unsigned int)sub_C444A0((__int64)&v103) > 0x40 )
          {
LABEL_62:
            *(_DWORD *)(v16 + 12) = 2;
            if ( v105 )
            {
              v105 = 0;
              sub_969240((__int64 *)&v103);
            }
            goto LABEL_16;
          }
          v21 = v81;
          v22 = (_QWORD *)*v103;
        }
        else
        {
          v22 = v103;
        }
        if ( v21 < (unsigned __int64)v22 )
          goto LABEL_62;
      }
      v23 = *(_DWORD *)(v16 + 12);
      if ( v23 != 1 )
      {
        if ( v23 == 2 )
          BUG();
        if ( v23 )
          goto LABEL_32;
        v99 = 1;
        v118 = v16;
        v117.m128i_i64[0] = (__int64)&v99;
        v110 = &v97;
        v117.m128i_i64[1] = v95;
        v119 = v3;
        v111 = v3;
        v112 = v95;
        if ( (unsigned __int8)sub_252FFB0(
                                v3,
                                (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2588E60,
                                (__int64)&v117,
                                v95,
                                *(_QWORD *)v16,
                                0,
                                1,
                                1,
                                (unsigned __int8 (__fastcall *)(__int64, __int64, __int64))sub_2537B70,
                                (__int64)&v110)
          && v99 )
        {
          goto LABEL_31;
        }
        *(_DWORD *)(v16 + 12) = 1;
      }
      if ( !v97 && !(unsigned __int8)sub_259B8C0(v3, v95, v90, 1, &v117, 0, 0) )
      {
LABEL_67:
        v20 = *(_DWORD *)(v16 + 8);
LABEL_58:
        *(_DWORD *)(v16 + 12) = 2;
        v93 = 0;
        goto LABEL_32;
      }
      if ( !v98 )
      {
        v53 = v107;
        *v106 = 1;
        v54 = *(_QWORD *)(v53 + 184);
        if ( v54 != v54 + 16LL * *(unsigned int *)(v53 + 192) )
        {
          v80 = v3;
          v86 = v16;
          v55 = v54 + 16LL * *(unsigned int *)(v53 + 192);
          v82 = v10;
          v56 = *(_QWORD *)(v53 + 184);
          v57 = v53;
          while ( 2 )
          {
            v58 = *(_QWORD *)(v56 + 8);
            if ( *(_BYTE *)(v58 + 16) )
              goto LABEL_94;
            LOBYTE(v110) = 0;
            if ( (unsigned __int8)sub_251BFD0(v108, *(_QWORD *)v58, v57, (_QWORD *)*v109, &v110, 1, 1, 0) )
              goto LABEL_94;
            v59 = sub_98ACB0(*(unsigned __int8 **)(v58 + 8), 6u);
            if ( v59 )
            {
              v61 = *v59;
              if ( (_BYTE)v61 == 20 || (unsigned int)(unsigned __int8)v61 - 12 <= 1 )
                goto LABEL_94;
              if ( (unsigned __int8)v61 > 0x1Cu )
              {
                v62 = (unsigned int)(v61 - 34);
                if ( (unsigned __int8)v62 <= 0x33u )
                {
                  v63 = 0x8000000000041LL;
                  if ( _bittest64(&v63, v62) )
                  {
                    v64 = *(unsigned int *)(v57 + 128);
                    v65 = *(_QWORD *)(v57 + 112);
                    v117.m128i_i64[0] = (__int64)v59;
                    if ( (_DWORD)v64 )
                    {
                      v66 = (unsigned int)(v64 - 1);
                      v67 = v66 & (((unsigned int)v59 >> 9) ^ ((unsigned int)v59 >> 4));
                      v68 = v65 + 16LL * v67;
                      v69 = *(_QWORD *)v68;
                      if ( v59 == *(unsigned __int8 **)v68 )
                      {
LABEL_108:
                        v70 = 16 * v64;
                        if ( v68 != v70 + v65
                          && *(_QWORD *)(*(_QWORD *)(v57 + 136) + 16LL * *(unsigned int *)(v68 + 8) + 8) )
                        {
                          sub_2575530(v58 + 24, v117.m128i_i64, v70, v68, v66, v60);
LABEL_94:
                          v56 += 16;
                          if ( v55 == v56 )
                          {
                            v16 = v86;
                            v10 = v82;
                            v3 = v80;
                            goto LABEL_57;
                          }
                          continue;
                        }
                      }
                      else
                      {
                        v71 = 1;
                        while ( v69 != -4096 )
                        {
                          v60 = (unsigned int)(v71 + 1);
                          v67 = v66 & (v71 + v67);
                          v68 = v65 + 16LL * v67;
                          v69 = *(_QWORD *)v68;
                          if ( v59 == *(unsigned __int8 **)v68 )
                            goto LABEL_108;
                          v71 = v60;
                        }
                      }
                    }
                  }
                }
              }
            }
            break;
          }
          *(_BYTE *)(v58 + 16) = 1;
          goto LABEL_94;
        }
      }
LABEL_57:
      v34 = *(_DWORD *)(v16 + 64);
      v20 = *(_DWORD *)(v16 + 8);
      if ( v34 != 1 )
        goto LABEL_58;
      v35 = **(_QWORD **)(v16 + 56);
      v36 = *(_QWORD *)(v95 + 160);
      v37 = *(unsigned int *)(v95 + 176);
      if ( !(_DWORD)v37 )
        goto LABEL_58;
      v38 = (v37 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v39 = (__int64 *)(v36 + 16LL * v38);
      v40 = *v39;
      if ( v35 != *v39 )
      {
        while ( v40 != -4096 )
        {
          v38 = (v37 - 1) & (v34 + v38);
          v39 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v39;
          if ( v35 == *v39 )
            goto LABEL_72;
          ++v34;
        }
        goto LABEL_58;
      }
LABEL_72:
      if ( v39 == (__int64 *)(v36 + 16 * v37) )
        goto LABEL_58;
      v41 = *(_QWORD *)(*(_QWORD *)(v95 + 184) + 16LL * *((unsigned int *)v39 + 2) + 8);
      if ( !v41 || *(_BYTE *)(v41 + 16) )
        goto LABEL_58;
      v42 = *(_DWORD *)(v41 + 64);
      if ( v42 )
      {
        if ( v42 != 1 )
          goto LABEL_58;
        v43 = *(_QWORD *)v16;
        if ( **(_QWORD **)(v41 + 56) != *(_QWORD *)v16 )
          goto LABEL_58;
        if ( v20 != 109 )
        {
          if ( *(_BYTE *)v43 != 34 )
          {
            v44 = *(_QWORD *)(v43 + 32);
            if ( v44 == *(_QWORD *)(v43 + 40) + 48LL || !v44 )
              v43 = 0;
            else
              v43 = v44 - 24;
          }
          v85 = **(_QWORD **)(v16 + 56);
          if ( v83 )
          {
            v45 = sub_2568740(v83, v43);
            sub_254C700((__int64)&v110, v45);
            sub_254C700((__int64)&v117, v83 + 200);
            v46 = v85;
            v47 = v85 & 0xFFFFFFFFFFFFFFFBLL;
            v48 = v85 | 4;
            if ( !v113 )
              goto LABEL_118;
            v49 = v113 - 1;
            v50 = 1;
            v51 = *(_QWORD *)(v111 + 8LL * ((v113 - 1) & ((unsigned int)v48 ^ (unsigned int)(v48 >> 9))));
            v52 = (v113 - 1) & (v48 ^ (v48 >> 9));
            if ( v48 != v51 )
            {
              v77 = 1;
              while ( v51 != -4 )
              {
                v52 = v49 & (v77 + v52);
                v51 = *(_QWORD *)(v111 + 8LL * v52);
                if ( v48 == v51 )
                  goto LABEL_86;
                ++v77;
              }
              v78 = v49 & (v47 ^ (v47 >> 9));
              v79 = *(_QWORD *)(v111 + 8LL * v78);
              if ( v79 != v47 )
              {
                while ( v79 != -4 )
                {
                  v78 = v49 & (v50 + v78);
                  v79 = *(_QWORD *)(v111 + 8LL * v78);
                  if ( v47 == v79 )
                    goto LABEL_86;
                  ++v50;
                }
LABEL_118:
                v87 = v16;
                v72 = v114;
                v73 = v46;
                do
                {
                  if ( v120 == v72 && v115 == v121 && v116 == v122 )
                  {
                    v16 = v87;
                    sub_C7D6A0(v117.m128i_i64[1], 8LL * (unsigned int)v119, 8);
                    sub_C7D6A0(v111, 8LL * v113, 8);
                    goto LABEL_67;
                  }
                  v72 = sub_3106C80(&v110);
                  v114 = v72;
                }
                while ( v73 != v72 );
                v16 = v87;
              }
            }
LABEL_86:
            sub_C7D6A0(v117.m128i_i64[1], 8LL * (unsigned int)v119, 8);
            sub_C7D6A0(v111, 8LL * v113, 8);
LABEL_31:
            v20 = *(_DWORD *)(v16 + 8);
            goto LABEL_32;
          }
          goto LABEL_58;
        }
        v24 = v105;
        if ( !*(_BYTE *)(v16 + 17) )
          goto LABEL_45;
        if ( v105 )
          goto LABEL_46;
LABEL_99:
        *(_BYTE *)(v16 + 17) = 0;
        if ( v24 )
          goto LABEL_46;
LABEL_17:
        v10 += 16;
        if ( v96 == v10 )
          return v93;
      }
      else
      {
LABEL_32:
        v24 = v105;
        if ( !*(_BYTE *)(v16 + 17) )
          goto LABEL_45;
        if ( !v105 )
        {
          *(_BYTE *)(v16 + 17) = 0;
          goto LABEL_17;
        }
        if ( v20 == 109 )
          goto LABEL_46;
        v25 = *(_QWORD *)(*(_QWORD *)v16 + 40LL);
        v26 = *(_QWORD *)(v89 + 80);
        if ( v26 )
          v26 -= 24;
        if ( v25 == v26 )
          goto LABEL_46;
        v24 = v92;
        if ( !v92 )
        {
          v92 = 1;
          v91 = sub_31052D0(v89, v88);
          v24 = v105;
        }
        if ( !v88 || v91 )
          goto LABEL_99;
        v27 = *(_DWORD *)(v88 + 24);
        v28 = *(_QWORD *)(v88 + 8);
        if ( !v27 )
          goto LABEL_45;
        v29 = v27 - 1;
        v30 = v29 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
        v31 = (__int64 *)(v28 + 16LL * v30);
        v32 = *v31;
        if ( v25 == *v31 )
        {
LABEL_44:
          if ( !v31[1] )
            goto LABEL_45;
          goto LABEL_99;
        }
        v74 = 1;
        while ( v32 != -4096 )
        {
          v75 = v74 + 1;
          v76 = v29 & (v30 + v74);
          v30 = v76;
          v31 = (__int64 *)(v28 + 16 * v76);
          v32 = *v31;
          if ( v25 == *v31 )
            goto LABEL_44;
          v74 = v75;
        }
LABEL_45:
        if ( !v24 )
          goto LABEL_17;
LABEL_46:
        v105 = 0;
        v10 += 16;
        sub_969240((__int64 *)&v103);
        if ( v96 == v10 )
          return v93;
      }
    }
    LOBYTE(v110) = 0;
    v12.m128i_i64[0] = sub_250D2C0(v17, 0);
    v117 = v12;
    v13 = sub_2527570(v3, &v117, v95, &v110);
    v102 = v14;
    v101 = v13;
    if ( (_BYTE)v14 )
    {
      if ( !v101 || *v101 != 17 )
      {
        LOBYTE(v118) = 0;
        *(_DWORD *)(v16 + 12) = 2;
        if ( (_BYTE)v118 )
        {
LABEL_49:
          LOBYTE(v118) = 0;
          sub_969240(v117.m128i_i64);
        }
LABEL_16:
        v93 = 0;
        goto LABEL_17;
      }
      v117.m128i_i32[2] = *((_DWORD *)v101 + 8);
      if ( v117.m128i_i32[2] > 0x40u )
        sub_C43780((__int64)&v117, (const void **)v101 + 3);
      else
        v117.m128i_i64[0] = *((_QWORD *)v101 + 3);
      v15 = v117.m128i_i32[2];
      LOBYTE(v118) = 1;
      if ( v117.m128i_i32[2] > 0x40u )
      {
        if ( v15 - (unsigned int)sub_C444A0((__int64)&v117) > 0x40
          || *(_QWORD *)v117.m128i_i64[0] > 0x100000000uLL
          || (unsigned int)sub_C44630((__int64)&v117) != 1 )
        {
          goto LABEL_15;
        }
LABEL_60:
        LOBYTE(v118) = 0;
        sub_969240(v117.m128i_i64);
        goto LABEL_20;
      }
    }
    else
    {
      v117.m128i_i32[2] = 64;
      v117.m128i_i64[0] = 0;
      LOBYTE(v118) = 1;
    }
    if ( (unsigned __int64)(v117.m128i_i64[0] - 1) > 0xFFFFFFFF || ((v117.m128i_i64[0] - 1) & v117.m128i_i64[0]) != 0 )
    {
LABEL_15:
      *(_DWORD *)(v16 + 12) = 2;
      if ( (_BYTE)v118 )
        goto LABEL_49;
      goto LABEL_16;
    }
    goto LABEL_60;
  }
  return 1;
}
