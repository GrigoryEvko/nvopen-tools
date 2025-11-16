// Function: sub_F8C4D0
// Address: 0xf8c4d0
//
unsigned __int8 *__fastcall sub_F8C4D0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int64 v10; // rbx
  __int64 *v11; // rax
  __int64 v12; // r15
  __int64 v13; // rbx
  int v14; // ecx
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // r10
  __int64 v18; // rsi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 *v22; // rdx
  __int64 v23; // r13
  __int64 v24; // r15
  _QWORD *v25; // rax
  __int64 v26; // rax
  unsigned int v27; // edi
  int v28; // edx
  __int64 v29; // rax
  __int64 v30; // r14
  int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  char v35; // dl
  __int64 v36; // rcx
  _QWORD *v37; // rax
  _QWORD *v38; // rdx
  unsigned __int64 v39; // rax
  int v40; // edx
  unsigned __int64 v41; // rax
  __int64 v42; // r9
  unsigned __int64 v43; // rax
  __int64 v44; // rsi
  __int64 *v45; // r8
  int v46; // eax
  int v47; // eax
  unsigned int v48; // edx
  unsigned __int64 v49; // rbx
  __int64 *v50; // r15
  _QWORD *v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdi
  __int64 v55; // rsi
  _QWORD *v56; // rax
  __int64 v58; // r9
  void *v59; // r10
  __int64 v60; // rax
  size_t v61; // r14
  __int64 v62; // r8
  __int64 v63; // rdi
  _QWORD *v64; // rax
  _QWORD *v65; // rax
  _QWORD *v66; // r15
  __int64 v67; // r14
  __int64 v68; // rax
  __int64 *v69; // rax
  _QWORD *v70; // r15
  __int64 *v71; // r13
  __int64 v72; // rax
  __int64 *v73; // rax
  __int64 *v74; // rdi
  _QWORD *v75; // r13
  __int64 v76; // rsi
  __int64 v77; // r9
  unsigned __int64 v78; // r8
  __int64 *v79; // rax
  __int64 *v80; // rdx
  __int64 *i; // rdx
  __int64 v82; // rax
  __int64 v83; // r15
  __int64 v84; // r13
  __int64 v85; // r14
  __int64 v86; // rbx
  _QWORD *v87; // rax
  _QWORD *v88; // rax
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // r15
  __int64 v92; // rax
  __int16 v93; // dx
  __int16 v94; // r14
  __int64 v95; // r13
  __int64 *v96; // rax
  _QWORD *v97; // r15
  __int64 *v98; // rdi
  int v99; // eax
  int v100; // eax
  __int64 v101; // rsi
  unsigned __int8 *v102; // rsi
  __int64 *v103; // r15
  _QWORD *v104; // rbx
  __int64 *v105; // r13
  __int64 v106; // rax
  __int64 v107; // rax
  __int16 v108; // r15
  _BYTE *v109; // r14
  _QWORD *v110; // rax
  __int64 v111; // [rsp+8h] [rbp-D8h]
  __int64 *v112; // [rsp+10h] [rbp-D0h]
  __int64 v113; // [rsp+18h] [rbp-C8h]
  _QWORD *srca; // [rsp+20h] [rbp-C0h]
  void *src; // [rsp+20h] [rbp-C0h]
  void *srcb; // [rsp+20h] [rbp-C0h]
  int v117; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v118; // [rsp+28h] [rbp-B8h]
  int v119; // [rsp+28h] [rbp-B8h]
  __int64 v120; // [rsp+30h] [rbp-B0h]
  __int64 v121; // [rsp+38h] [rbp-A8h]
  __int64 v122; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v123; // [rsp+38h] [rbp-A8h]
  __int64 v124[2]; // [rsp+40h] [rbp-A0h] BYREF
  _QWORD v125[2]; // [rsp+50h] [rbp-90h] BYREF
  char v126; // [rsp+60h] [rbp-80h]
  char v127; // [rsp+61h] [rbp-7Fh]
  char *v128; // [rsp+70h] [rbp-70h] BYREF
  __int64 v129; // [rsp+78h] [rbp-68h]
  __int64 v130; // [rsp+80h] [rbp-60h] BYREF
  _QWORD *v131; // [rsp+88h] [rbp-58h]
  char v132; // [rsp+90h] [rbp-50h] BYREF
  char v133; // [rsp+91h] [rbp-4Fh]

  v2 = a2;
  v4 = *a1;
  v5 = sub_D95540(**(_QWORD **)(a2 + 32));
  v6 = sub_D97090(v4, v5);
  v7 = 0;
  v120 = v6;
  v121 = *(_QWORD *)(v2 + 48);
  if ( (_BYTE)qword_4F8C1C8 && *((_BYTE *)a1 + 512) )
    v7 = v6;
  v8 = sub_D48760(v121, v7);
  v9 = v8;
  if ( v8 && (v10 = sub_D97050(*a1, *(_QWORD *)(v8 + 8)), v10 >= sub_D97050(*a1, v120)) )
  {
    v49 = sub_D97050(*a1, *(_QWORD *)(v9 + 8));
    if ( v49 > sub_D97050(*a1, v120) && *(_BYTE *)(sub_D95540(**(_QWORD **)(v2 + 32)) + 8) != 14 )
    {
      v78 = *(_QWORD *)(v2 + 40);
      v128 = (char *)&v130;
      v129 = 0x400000000LL;
      if ( v78 )
      {
        v79 = &v130;
        v80 = &v130;
        if ( v78 > 4 )
        {
          v123 = v78;
          sub_C8D5F0((__int64)&v128, &v130, v78, 8u, v78, v77);
          v80 = (__int64 *)v128;
          v78 = v123;
          v79 = (__int64 *)&v128[8 * (unsigned int)v129];
        }
        for ( i = &v80[v78]; i != v79; ++v79 )
        {
          if ( v79 )
            *v79 = 0;
        }
        v82 = *(_QWORD *)(v2 + 40);
        LODWORD(v129) = v78;
        if ( (_DWORD)v82 )
        {
          v83 = v2;
          v84 = v9;
          v85 = 0;
          v86 = 8LL * (unsigned int)v82;
          do
          {
            v87 = sub_DC5890(*a1, *(_QWORD *)(*(_QWORD *)(v83 + 32) + v85), *(_QWORD *)(v84 + 8));
            *(_QWORD *)&v128[v85] = v87;
            v85 += 8;
          }
          while ( v86 != v85 );
          v2 = v83;
        }
      }
      v88 = sub_DBFF60(*a1, (unsigned int *)&v128, *(_QWORD *)(v2 + 48), *(_WORD *)(v2 + 28) & 1);
      v89 = sub_F894B0((__int64)a1, (__int64)v88);
      v90 = a1[72];
      v91 = v89;
      if ( v90 )
        v90 -= 24;
      v92 = sub_F7D460((__int64)a1, v89, v90);
      v94 = v93;
      v95 = v92;
      v122 = *a1;
      v96 = sub_DA3860((_QWORD *)*a1, v91);
      v97 = sub_DC5200(v122, (__int64)v96, v120, 0);
      if ( !v95 )
        BUG();
      sub_A88F30((__int64)(a1 + 65), *(_QWORD *)(v95 + 16), v95, v94);
      v76 = (__int64)v97;
      goto LABEL_76;
    }
    if ( sub_D968A0(**(_QWORD **)(v2 + 32)) )
    {
LABEL_65:
      if ( *(_QWORD *)(v2 + 40) != 2 )
        goto LABEL_66;
      if ( !sub_D96900(*(_QWORD *)(*(_QWORD *)(v2 + 32) + 8LL)) )
      {
        v103 = (__int64 *)*a1;
        if ( *(_QWORD *)(v2 + 40) == 2 )
        {
          v104 = sub_DD2CB0(*a1, *(_QWORD *)(*(_QWORD *)(v2 + 32) + 8LL), *(_QWORD *)(v9 + 8));
          v130 = (__int64)sub_DA3860((_QWORD *)*a1, v9);
          v128 = (char *)&v130;
          v131 = v104;
          v129 = 0x200000002LL;
          v105 = sub_DC8BD0(v103, (__int64)&v128, 0, 0);
          if ( v128 != (char *)&v130 )
            _libc_free(v128, &v128);
          v53 = v120;
          v55 = (__int64)v105;
          v54 = (__int64)v103;
          goto LABEL_69;
        }
LABEL_66:
        v50 = sub_DA3860((_QWORD *)*a1, v9);
        v51 = sub_DD2CB0(*a1, v2, *(_QWORD *)(v9 + 8));
        if ( *((_WORD *)v51 + 12) == 8 )
          v2 = (__int64)v51;
        v52 = sub_DD0540(v2, (__int64)v50, (__int64 *)*a1);
        v53 = v120;
        v54 = *a1;
        v55 = (__int64)v52;
LABEL_69:
        v56 = sub_DC5820(v54, v55, v53);
        return (unsigned __int8 *)sub_F894B0((__int64)a1, (__int64)v56);
      }
      return (unsigned __int8 *)v9;
    }
  }
  else if ( sub_D968A0(**(_QWORD **)(v2 + 32)) )
  {
    v11 = *(__int64 **)(v121 + 32);
    v12 = *v11;
    v13 = *(_QWORD *)(*v11 + 16);
    if ( v13 )
    {
      while ( (unsigned __int8)(**(_BYTE **)(v13 + 24) - 30) > 0xAu )
      {
        v13 = *(_QWORD *)(v13 + 8);
        if ( !v13 )
          goto LABEL_62;
      }
      v133 = 1;
      v14 = 0;
      v132 = 3;
      v128 = "indvar";
      v15 = v13;
      while ( 1 )
      {
        v15 = *(_QWORD *)(v15 + 8);
        if ( !v15 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v15 + 24) - 30) <= 0xAu )
        {
          v15 = *(_QWORD *)(v15 + 8);
          ++v14;
          if ( !v15 )
            goto LABEL_13;
        }
      }
LABEL_13:
      v117 = v14 + 1;
    }
    else
    {
LABEL_62:
      v133 = 1;
      v13 = 0;
      v128 = "indvar";
      v132 = 3;
      v117 = 0;
    }
    v16 = sub_BD2DA0(80);
    v9 = v16;
    if ( v16 )
    {
      srca = (_QWORD *)v16;
      sub_B44260(v16, v120, 55, 0x8000000u, 0, 0);
      *(_DWORD *)(v9 + 72) = v117;
      sub_BD6B50((unsigned __int8 *)v9, (const char **)&v128);
      sub_BD2A10(v9, *(_DWORD *)(v9 + 72), 1);
      v17 = srca;
    }
    else
    {
      v17 = 0;
    }
    sub_B44220(v17, *(_QWORD *)(v12 + 56), 1);
    sub_F86EA0((__int64)a1, v9);
    v18 = 1;
    v128 = 0;
    v129 = (__int64)&v132;
    v130 = 4;
    LODWORD(v131) = 0;
    BYTE4(v131) = 1;
    v111 = sub_AD64C0(v120, 1, 0);
    if ( !v13 )
      goto LABEL_98;
    v22 = *(__int64 **)(v13 + 24);
    src = (void *)v2;
    v23 = v9;
    v24 = v22[5];
    if ( !BYTE4(v131) )
      goto LABEL_40;
LABEL_18:
    v25 = (_QWORD *)v129;
    v19 = HIDWORD(v130);
    v22 = (__int64 *)(v129 + 8LL * HIDWORD(v130));
    if ( (__int64 *)v129 == v22 )
    {
LABEL_100:
      if ( HIDWORD(v130) < (unsigned int)v130 )
      {
        ++HIDWORD(v130);
        v36 = v121;
        *v22 = v24;
        ++v128;
        if ( !*(_BYTE *)(v121 + 84) )
        {
LABEL_102:
          v18 = v24;
          if ( !sub_C8CA60(v121 + 56, v24) )
          {
LABEL_103:
            v30 = sub_AD6530(v120, v18);
            v99 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
            if ( v99 == *(_DWORD *)(v23 + 72) )
            {
              sub_B48D90(v23);
              v99 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
            }
            v100 = (v99 + 1) & 0x7FFFFFF;
            v19 = v100 | *(_DWORD *)(v23 + 4) & 0xF8000000;
            v32 = (__int64 *)(*(_QWORD *)(v23 - 8) + 32LL * (unsigned int)(v100 - 1));
            *(_DWORD *)(v23 + 4) = v19;
            if ( *v32 )
            {
              v18 = v32[2];
              v19 = v32[1];
              *(_QWORD *)v18 = v19;
              if ( v19 )
              {
                v18 = v32[2];
                *(_QWORD *)(v19 + 16) = v18;
              }
            }
            *v32 = v30;
            if ( !v30 )
            {
LABEL_36:
              *(_QWORD *)(*(_QWORD *)(v23 - 8)
                        + 32LL * *(unsigned int *)(v23 + 72)
                        + 8LL * ((*(_DWORD *)(v23 + 4) & 0x7FFFFFFu) - 1)) = v24;
              while ( 1 )
              {
                v13 = *(_QWORD *)(v13 + 8);
                if ( !v13 )
                  break;
                v22 = *(__int64 **)(v13 + 24);
                if ( (unsigned __int8)(*(_BYTE *)v22 - 30) <= 0xAu )
                {
                  v24 = v22[5];
                  if ( !BYTE4(v131) )
                    goto LABEL_40;
                  goto LABEL_18;
                }
              }
              v9 = v23;
              v2 = (__int64)src;
LABEL_98:
              if ( !BYTE4(v131) )
                _libc_free(v129, v18);
              goto LABEL_65;
            }
            v19 = *(_QWORD *)(v30 + 16);
            v18 = v30 + 16;
            v32[1] = v19;
            if ( v19 )
              *(_QWORD *)(v19 + 16) = v32 + 1;
LABEL_35:
            v32[2] = v18;
            *(_QWORD *)(v30 + 16) = v32;
            goto LABEL_36;
          }
LABEL_46:
          v39 = *(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v24 + 48 == v39 )
          {
            v41 = 0;
          }
          else
          {
            if ( !v39 )
              goto LABEL_130;
            v40 = *(unsigned __int8 *)(v39 - 24);
            v41 = v39 - 24;
            if ( (unsigned int)(v40 - 30) >= 0xB )
              v41 = 0;
          }
          v127 = 1;
          v42 = v113;
          v126 = 3;
          v124[0] = (__int64)"indvar.next";
          LOWORD(v42) = 0;
          v30 = sub_B504D0(13, v23, v111, (__int64)v124, v41 + 24, v42);
          v43 = *(_QWORD *)(v24 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v43 == v24 + 48 )
            goto LABEL_129;
          if ( !v43 )
LABEL_130:
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v43 - 24) - 30 > 0xA )
LABEL_129:
            BUG();
          v44 = *(_QWORD *)(v43 + 24);
          v45 = (__int64 *)(v30 + 48);
          v124[0] = v44;
          if ( v44 )
          {
            sub_B96E90((__int64)v124, v44, 1);
            v45 = (__int64 *)(v30 + 48);
            if ( (__int64 *)(v30 + 48) == v124 )
            {
              if ( v124[0] )
                sub_B91220((__int64)v124, v124[0]);
              goto LABEL_57;
            }
            v101 = *(_QWORD *)(v30 + 48);
            if ( !v101 )
            {
LABEL_114:
              v102 = (unsigned __int8 *)v124[0];
              *(_QWORD *)(v30 + 48) = v124[0];
              if ( v102 )
                sub_B976B0((__int64)v124, v102, (__int64)v45);
LABEL_57:
              v18 = v30;
              sub_F86EA0((__int64)a1, v30);
              v46 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
              if ( v46 == *(_DWORD *)(v23 + 72) )
              {
                sub_B48D90(v23);
                v46 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
              }
              v47 = (v46 + 1) & 0x7FFFFFF;
              v48 = v47 | *(_DWORD *)(v23 + 4) & 0xF8000000;
              v32 = (__int64 *)(*(_QWORD *)(v23 - 8) + 32LL * (unsigned int)(v47 - 1));
              *(_DWORD *)(v23 + 4) = v48;
              if ( !*v32 )
                goto LABEL_32;
              goto LABEL_30;
            }
          }
          else
          {
            if ( v45 == v124 )
              goto LABEL_57;
            v101 = *(_QWORD *)(v30 + 48);
            if ( !v101 )
              goto LABEL_57;
          }
          v112 = v45;
          sub_B91220((__int64)v45, v101);
          v45 = v112;
          goto LABEL_114;
        }
LABEL_42:
        v37 = *(_QWORD **)(v36 + 64);
        v38 = &v37[*(unsigned int *)(v36 + 76)];
        if ( v37 == v38 )
          goto LABEL_103;
        while ( v24 != *v37 )
        {
          if ( v38 == ++v37 )
            goto LABEL_103;
        }
        goto LABEL_46;
      }
LABEL_40:
      v18 = v24;
      sub_C8CC70((__int64)&v128, v24, (__int64)v22, v19, v20, v21);
      if ( v35 )
      {
        v36 = v121;
        if ( !*(_BYTE *)(v121 + 84) )
          goto LABEL_102;
        goto LABEL_42;
      }
    }
    else
    {
      while ( v24 != *v25 )
      {
        if ( v22 == ++v25 )
          goto LABEL_100;
      }
    }
    v18 = *(_QWORD *)(v23 - 8);
    v26 = 0x1FFFFFFFE0LL;
    v27 = *(_DWORD *)(v23 + 72);
    v28 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
    if ( v28 )
    {
      v29 = 0;
      v19 = v18 + 32LL * v27;
      do
      {
        if ( v24 == *(_QWORD *)(v19 + 8 * v29) )
        {
          v26 = 32 * v29;
          goto LABEL_27;
        }
        ++v29;
      }
      while ( v28 != (_DWORD)v29 );
      v26 = 0x1FFFFFFFE0LL;
    }
LABEL_27:
    v30 = *(_QWORD *)(v18 + v26);
    if ( v28 == v27 )
    {
      sub_B48D90(v23);
      v18 = *(_QWORD *)(v23 - 8);
      v28 = *(_DWORD *)(v23 + 4) & 0x7FFFFFF;
    }
    v31 = (v28 + 1) & 0x7FFFFFF;
    *(_DWORD *)(v23 + 4) = v31 | *(_DWORD *)(v23 + 4) & 0xF8000000;
    v32 = (__int64 *)(v18 + 32LL * (unsigned int)(v31 - 1));
    if ( !*v32 )
      goto LABEL_32;
LABEL_30:
    v18 = v32[2];
    v33 = v32[1];
    *(_QWORD *)v18 = v33;
    if ( v33 )
    {
      v18 = v32[2];
      *(_QWORD *)(v33 + 16) = v18;
    }
LABEL_32:
    *v32 = v30;
    if ( !v30 )
      goto LABEL_36;
    v34 = *(_QWORD *)(v30 + 16);
    v18 = v30 + 16;
    v32[1] = v34;
    if ( v34 )
      *(_QWORD *)(v34 + 16) = v32 + 1;
    goto LABEL_35;
  }
  if ( *(_BYTE *)(sub_D95540(**(_QWORD **)(v2 + 32)) + 8) != 14 )
  {
    v59 = *(void **)(v2 + 32);
    v129 = 0x400000000LL;
    v60 = *(_QWORD *)(v2 + 40);
    v128 = (char *)&v130;
    v61 = 8 * v60;
    v62 = (8 * v60) >> 3;
    if ( (unsigned __int64)(8 * v60) > 0x20 )
    {
      srcb = v59;
      v118 = (8 * v60) >> 3;
      sub_C8D5F0((__int64)&v128, &v130, v118, 8u, v62, v58);
      LODWORD(v62) = v118;
      v59 = srcb;
      v98 = (__int64 *)&v128[8 * (unsigned int)v129];
    }
    else
    {
      if ( !v61 )
      {
LABEL_73:
        v63 = *a1;
        LODWORD(v129) = v61 + v62;
        v64 = sub_DA2C50(v63, v120, 0, 0);
        *(_QWORD *)v128 = v64;
        v65 = sub_DBFF60(*a1, (unsigned int *)&v128, v121, *(_WORD *)(v2 + 28) & 1);
        v66 = (_QWORD *)*a1;
        v67 = (__int64)v65;
        v68 = sub_F894B0((__int64)a1, **(_QWORD **)(v2 + 32));
        v69 = sub_DA3860(v66, v68);
        v70 = (_QWORD *)*a1;
        v71 = v69;
        v72 = sub_F894B0((__int64)a1, v67);
        v73 = sub_DA3860(v70, v72);
        v74 = (__int64 *)*a1;
        v125[1] = v73;
        v125[0] = v71;
        v124[0] = (__int64)v125;
        v124[1] = 0x200000002LL;
        v75 = sub_DC7EB0(v74, (__int64)v124, 0, 0);
        if ( (_QWORD *)v124[0] != v125 )
          _libc_free(v124[0], v124);
        v76 = (__int64)v75;
LABEL_76:
        v9 = sub_F894B0((__int64)a1, v76);
        if ( v128 != (char *)&v130 )
          _libc_free(v128, v76);
        return (unsigned __int8 *)v9;
      }
      v98 = &v130;
    }
    v119 = v62;
    memcpy(v98, v59, v61);
    LODWORD(v61) = v129;
    LODWORD(v62) = v119;
    goto LABEL_73;
  }
  v106 = sub_D97190(*a1, v2);
  v107 = sub_F894B0((__int64)a1, v106);
  v108 = *(_WORD *)(v2 + 28);
  v109 = (_BYTE *)v107;
  v110 = sub_DCB010((__int64 *)*a1, v2);
  return sub_F8A290((__int64)a1, (__int64)v110, v109, v108 & 2);
}
