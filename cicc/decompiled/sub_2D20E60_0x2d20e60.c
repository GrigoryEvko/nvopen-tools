// Function: sub_2D20E60
// Address: 0x2d20e60
//
void __fastcall sub_2D20E60(__int64 a1)
{
  __int64 v1; // r15
  __int64 v2; // r13
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int *v6; // rax
  __int64 v7; // rdx
  unsigned __int8 *v8; // r15
  unsigned __int8 v9; // dl
  __int64 v10; // rdx
  _BYTE *v11; // rsi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 i; // rbx
  unsigned __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // r13
  __int64 *v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rsi
  __int64 v22; // r9
  __int64 v23; // r8
  unsigned int *v24; // rax
  int v25; // ecx
  unsigned int *v26; // rdx
  __int64 v27; // r9
  __int64 v28; // r8
  unsigned int *v29; // rax
  int v30; // ecx
  unsigned int *v31; // rdx
  bool v32; // al
  unsigned __int64 v33; // rsi
  __int64 v34; // r14
  __int64 v35; // r15
  __int64 v36; // rax
  char v37; // al
  __int16 v38; // cx
  _QWORD *v39; // rax
  __int64 v40; // r9
  __int64 v41; // r13
  unsigned int *v42; // r14
  unsigned int *v43; // r12
  __int64 v44; // rdx
  unsigned int v45; // esi
  _BYTE *v46; // rsi
  _QWORD *v47; // rdx
  _QWORD *v48; // rax
  __int64 **v49; // r14
  int v50; // eax
  unsigned __int64 v51; // rsi
  unsigned __int64 v52; // r14
  __int64 **v53; // r13
  unsigned int v54; // esi
  unsigned __int64 v55; // r14
  __int64 v56; // r15
  __int64 v57; // rax
  char v58; // al
  __int16 v59; // cx
  __int64 v60; // r9
  _QWORD *v61; // r13
  unsigned int *v62; // r14
  unsigned int *v63; // r12
  __int64 v64; // rdx
  unsigned int v65; // esi
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rdi
  __int64 v70; // rax
  __int64 v71; // rbx
  __int64 v72; // r12
  unsigned __int64 v73; // rsi
  unsigned __int64 v74; // rax
  unsigned __int64 v75; // rsi
  unsigned __int64 v76; // rax
  __int64 v77; // [rsp-8h] [rbp-1A8h]
  __int64 v78; // [rsp+0h] [rbp-1A0h]
  __int64 v79; // [rsp+8h] [rbp-198h]
  unsigned __int64 v80; // [rsp+10h] [rbp-190h]
  unsigned __int64 v81; // [rsp+10h] [rbp-190h]
  unsigned __int64 v82; // [rsp+10h] [rbp-190h]
  int v83; // [rsp+18h] [rbp-188h]
  int v84; // [rsp+18h] [rbp-188h]
  __int64 v85; // [rsp+18h] [rbp-188h]
  __int64 v86; // [rsp+18h] [rbp-188h]
  __int64 v87; // [rsp+20h] [rbp-180h]
  __int16 v88; // [rsp+44h] [rbp-15Ch]
  __int16 v89; // [rsp+46h] [rbp-15Ah]
  unsigned __int64 v90; // [rsp+50h] [rbp-150h] BYREF
  __int64 v91; // [rsp+58h] [rbp-148h] BYREF
  unsigned __int64 v92; // [rsp+60h] [rbp-140h] BYREF
  int v93; // [rsp+68h] [rbp-138h]
  unsigned __int64 v94; // [rsp+70h] [rbp-130h] BYREF
  _BYTE *v95; // [rsp+78h] [rbp-128h]
  _BYTE *v96; // [rsp+80h] [rbp-120h]
  unsigned __int64 v97; // [rsp+90h] [rbp-110h] BYREF
  _BYTE *v98; // [rsp+98h] [rbp-108h]
  _BYTE *v99; // [rsp+A0h] [rbp-100h]
  __int64 v100[4]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v101; // [rsp+D0h] [rbp-D0h]
  unsigned int *v102; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v103; // [rsp+E8h] [rbp-B8h]
  _BYTE v104[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v105; // [rsp+110h] [rbp-90h]
  __int64 v106; // [rsp+118h] [rbp-88h]
  __int64 v107; // [rsp+120h] [rbp-80h]
  _QWORD *v108; // [rsp+128h] [rbp-78h]
  void **v109; // [rsp+130h] [rbp-70h]
  void **v110; // [rsp+138h] [rbp-68h]
  __int64 v111; // [rsp+140h] [rbp-60h]
  int v112; // [rsp+148h] [rbp-58h]
  __int16 v113; // [rsp+14Ch] [rbp-54h]
  char v114; // [rsp+14Eh] [rbp-52h]
  __int64 v115; // [rsp+150h] [rbp-50h]
  __int64 v116; // [rsp+158h] [rbp-48h]
  void *v117; // [rsp+160h] [rbp-40h] BYREF
  void *v118; // [rsp+168h] [rbp-38h] BYREF

  v1 = *(_QWORD *)(a1 + 80);
  v94 = 0;
  v95 = 0;
  v96 = 0;
  if ( !v1 )
    BUG();
  v2 = *(_QWORD *)(v1 + 32);
  v3 = v1 + 24;
  if ( v2 != v3 )
  {
    v4 = v3;
    do
    {
      if ( !v2 )
        BUG();
      if ( *(_BYTE *)(v2 - 24) == 60 )
      {
        v5 = *(_QWORD *)(v2 - 8);
        if ( v5 )
        {
          while ( 1 )
          {
            v6 = *(unsigned int **)(v5 + 24);
            if ( *(_BYTE *)v6 != 62 )
              goto LABEL_8;
            v102 = *(unsigned int **)(v5 + 24);
            v7 = *((_QWORD *)v6 - 4);
            if ( v2 - 24 != v7 )
              goto LABEL_8;
            if ( !v7 )
              goto LABEL_8;
            v8 = (unsigned __int8 *)*((_QWORD *)v6 - 8);
            if ( !v8 )
              goto LABEL_8;
            v9 = *v8;
            if ( *v8 > 0x15u )
              break;
            v10 = *(_QWORD *)(v2 - 8);
            if ( !v10 )
            {
              v11 = v95;
              if ( v95 != v96 )
                goto LABEL_16;
LABEL_105:
              sub_278FF40((__int64)&v94, v11, &v102);
              goto LABEL_8;
            }
            if ( *(_QWORD *)(v10 + 8) )
            {
              v11 = v95;
              if ( v95 == v96 )
                goto LABEL_105;
LABEL_16:
              if ( v11 )
                goto LABEL_17;
LABEL_18:
              v95 = v11 + 8;
              v5 = *(_QWORD *)(v5 + 8);
              if ( !v5 )
                goto LABEL_19;
            }
            else
            {
LABEL_8:
              v5 = *(_QWORD *)(v5 + 8);
              if ( !v5 )
                goto LABEL_19;
            }
          }
          if ( v9 > 0x1Cu )
          {
            v66 = *((_QWORD *)v6 - 8);
            while ( 1 )
            {
              if ( v9 == 60 )
                goto LABEL_100;
              if ( v9 == 78 )
              {
                v66 = *(_QWORD *)(v66 - 32);
              }
              else
              {
                if ( v9 != 63 || !(unsigned __int8)sub_B4DCF0(v66) )
                {
LABEL_94:
                  v9 = *v8;
                  break;
                }
                v66 = *(_QWORD *)(v66 - 32LL * (*(_DWORD *)(v66 + 4) & 0x7FFFFFF));
              }
              v9 = *(_BYTE *)v66;
              if ( *(_BYTE *)v66 <= 0x1Cu )
                goto LABEL_94;
            }
          }
          if ( v9 != 61 )
          {
            v67 = *((_QWORD *)v8 + 2);
            if ( !v67 )
              goto LABEL_8;
            if ( *(_QWORD *)(v67 + 8) )
              goto LABEL_8;
            v68 = *(_QWORD *)(v2 - 8);
            if ( !v68 || *(_QWORD *)(v68 + 8) )
              goto LABEL_8;
          }
LABEL_100:
          v11 = v95;
          if ( v95 != v96 )
          {
            if ( !v95 )
              goto LABEL_18;
            v6 = v102;
LABEL_17:
            *(_QWORD *)v11 = v6;
            v11 = v95;
            goto LABEL_18;
          }
          goto LABEL_105;
        }
      }
LABEL_19:
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( v4 != v2 );
    v12 = v94;
    v97 = 0;
    v98 = 0;
    v99 = 0;
    v13 = (__int64)&v95[-v94] >> 3;
    if ( !(_DWORD)v13 )
      goto LABEL_113;
    v87 = 8LL * (unsigned int)(v13 - 1);
    for ( i = 0; ; i += 8 )
    {
      v15 = *(_QWORD *)(*(_QWORD *)(v12 + i) - 64LL);
      v16 = *(_QWORD *)(v15 + 8);
      v90 = v15;
      if ( (unsigned __int8)(*(_BYTE *)(v16 + 8) - 2) > 1u
        && !sub_BCAC40(v16, 8)
        && !sub_BCAC40(v16, 16)
        && !sub_BCAC40(v16, 32)
        && !sub_BCAC40(v16, 64)
        && *(_BYTE *)(v16 + 8) != 14 )
      {
        goto LABEL_58;
      }
      v102 = (unsigned int *)v16;
      if ( sub_BCAC40(v16, 8) )
      {
        v48 = (_QWORD *)sub_B2BE50(a1);
        v102 = (unsigned int *)sub_BCB2C0(v48);
      }
      v17 = sub_B6E160(*(__int64 **)(a1 + 40), 0x23BEu, (__int64)&v102, 1);
      v18 = (__int64 *)(i + v94);
      v19 = *(_QWORD *)(*(_QWORD *)(i + v94) + 48LL);
      v91 = v19;
      if ( v19 )
      {
        sub_B96E90((__int64)&v91, v19, 1);
        v18 = (__int64 *)(i + v94);
      }
      v20 = *v18;
      v108 = (_QWORD *)sub_BD5C60(*v18);
      v109 = &v117;
      v110 = &v118;
      v102 = (unsigned int *)v104;
      v113 = 512;
      v117 = &unk_49DA100;
      v103 = 0x200000000LL;
      v105 = 0;
      v106 = 0;
      v111 = 0;
      v112 = 0;
      v114 = 7;
      v115 = 0;
      v116 = 0;
      LOWORD(v107) = 0;
      v118 = &unk_49DA0B0;
      v105 = *(_QWORD *)(v20 + 40);
      v106 = v20 + 24;
      v21 = *(_QWORD *)sub_B46C60(v20);
      v100[0] = v21;
      if ( !v21 )
        break;
      sub_B96E90((__int64)v100, v21, 1);
      v23 = v100[0];
      if ( !v100[0] )
        break;
      v24 = v102;
      v25 = v103;
      v26 = &v102[4 * (unsigned int)v103];
      if ( v102 == v26 )
      {
LABEL_66:
        if ( (unsigned int)v103 >= (unsigned __int64)HIDWORD(v103) )
        {
          v75 = (unsigned int)v103 + 1LL;
          v76 = v78 & 0xFFFFFFFF00000000LL;
          v78 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v103) < v75 )
          {
            v82 = v76;
            v86 = v100[0];
            sub_C8D5F0((__int64)&v102, v104, v75, 0x10u, v100[0], v22);
            v76 = v82;
            v23 = v86;
            v26 = &v102[4 * (unsigned int)v103];
          }
          *(_QWORD *)v26 = v76;
          *((_QWORD *)v26 + 1) = v23;
          v23 = v100[0];
          LODWORD(v103) = v103 + 1;
        }
        else
        {
          if ( v26 )
          {
            *v26 = 0;
            *((_QWORD *)v26 + 1) = v23;
            v25 = v103;
            v23 = v100[0];
          }
          LODWORD(v103) = v25 + 1;
        }
LABEL_64:
        if ( !v23 )
          goto LABEL_35;
        goto LABEL_34;
      }
      while ( 1 )
      {
        v22 = *v24;
        if ( !(_DWORD)v22 )
          break;
        v24 += 4;
        if ( v26 == v24 )
          goto LABEL_66;
      }
      *((_QWORD *)v24 + 1) = v100[0];
LABEL_34:
      sub_B91220((__int64)v100, v23);
LABEL_35:
      v100[0] = v91;
      if ( !v91 || (sub_B96E90((__int64)v100, v91, 1), (v28 = v100[0]) == 0) )
      {
        sub_93FB40((__int64)&v102, 0);
        v28 = v100[0];
        goto LABEL_61;
      }
      v29 = v102;
      v30 = v103;
      v31 = &v102[4 * (unsigned int)v103];
      if ( v102 == v31 )
      {
LABEL_70:
        if ( (unsigned int)v103 >= (unsigned __int64)HIDWORD(v103) )
        {
          v73 = (unsigned int)v103 + 1LL;
          v74 = v79 & 0xFFFFFFFF00000000LL;
          v79 &= 0xFFFFFFFF00000000LL;
          if ( HIDWORD(v103) < v73 )
          {
            v81 = v74;
            v85 = v100[0];
            sub_C8D5F0((__int64)&v102, v104, v73, 0x10u, v100[0], v27);
            v74 = v81;
            v28 = v85;
            v31 = &v102[4 * (unsigned int)v103];
          }
          *(_QWORD *)v31 = v74;
          *((_QWORD *)v31 + 1) = v28;
          v28 = v100[0];
          LODWORD(v103) = v103 + 1;
        }
        else
        {
          if ( v31 )
          {
            *v31 = 0;
            *((_QWORD *)v31 + 1) = v28;
            v30 = v103;
            v28 = v100[0];
          }
          LODWORD(v103) = v30 + 1;
        }
LABEL_61:
        if ( !v28 )
          goto LABEL_43;
        goto LABEL_42;
      }
      while ( *v29 )
      {
        v29 += 4;
        if ( v31 == v29 )
          goto LABEL_70;
      }
      *((_QWORD *)v29 + 1) = v100[0];
LABEL_42:
      sub_B91220((__int64)v100, v28);
LABEL_43:
      v32 = sub_BCAC40(v16, 8);
      HIBYTE(v101) = 1;
      if ( v32 )
      {
        LOBYTE(v101) = 3;
        v100[0] = (__int64)"cast";
        v49 = (__int64 **)sub_BCB2C0(v108);
        v80 = v90;
        v83 = sub_BCB060(*(_QWORD *)(v90 + 8));
        v50 = sub_BCB060((__int64)v49);
        v51 = 0;
        v92 = sub_2D20CC0((__int64 *)&v102, 9 * (unsigned int)(v83 == v50) + 40, v80, v49, (__int64)v100, 0, v93, 0);
        v100[0] = (__int64)"move";
        v101 = 259;
        if ( v17 )
          v51 = *(_QWORD *)(v17 + 24);
        v52 = sub_921880(&v102, v51, v17, (int)&v92, 1, (__int64)v100, 0);
        v101 = 259;
        v100[0] = (__int64)"cast";
        v53 = (__int64 **)sub_BCB2B0(v108);
        v84 = sub_BCB060(*(_QWORD *)(v52 + 8));
        v54 = 49;
        if ( v84 != (unsigned int)sub_BCB060((__int64)v53) )
          v54 = 38;
        v55 = sub_2D20CC0((__int64 *)&v102, v54, v52, v53, (__int64)v100, 0, v93, 0);
        v56 = *(_QWORD *)(*(_QWORD *)(v94 + i) - 32LL);
        v57 = sub_AA4E30(v105);
        v58 = sub_AE5020(v57, *(_QWORD *)(v55 + 8));
        HIBYTE(v59) = HIBYTE(v88);
        v101 = 257;
        LOBYTE(v59) = v58;
        v88 = v59;
        v61 = sub_BD2C40(80, unk_3F10A10);
        if ( v61 )
        {
          sub_B4D3C0((__int64)v61, v55, v56, 0, v88, v60, 0, 0);
          v60 = v77;
        }
        (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64, __int64, __int64))*v110 + 2))(
          v110,
          v61,
          v100,
          v106,
          v107,
          v60);
        v62 = v102;
        v63 = &v102[4 * (unsigned int)v103];
        if ( v102 != v63 )
        {
          do
          {
            v64 = *((_QWORD *)v62 + 1);
            v65 = *v62;
            v62 += 4;
            sub_B99FD0((__int64)v61, v65, v64);
          }
          while ( v63 != v62 );
        }
      }
      else
      {
        LOBYTE(v101) = 3;
        v33 = 0;
        v100[0] = (__int64)"move";
        if ( v17 )
          v33 = *(_QWORD *)(v17 + 24);
        v34 = sub_921880(&v102, v33, v17, (int)&v90, 1, (__int64)v100, 0);
        v35 = *(_QWORD *)(*(_QWORD *)(v94 + i) - 32LL);
        v36 = sub_AA4E30(v105);
        v37 = sub_AE5020(v36, *(_QWORD *)(v34 + 8));
        HIBYTE(v38) = HIBYTE(v89);
        LOBYTE(v38) = v37;
        v89 = v38;
        v101 = 257;
        v39 = sub_BD2C40(80, unk_3F10A10);
        v41 = (__int64)v39;
        if ( v39 )
          sub_B4D3C0((__int64)v39, v34, v35, 0, v89, v40, 0, 0);
        (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v110 + 2))(v110, v41, v100, v106, v107);
        v42 = v102;
        v43 = &v102[4 * (unsigned int)v103];
        if ( v102 != v43 )
        {
          do
          {
            v44 = *((_QWORD *)v42 + 1);
            v45 = *v42;
            v42 += 4;
            sub_B99FD0(v41, v45, v44);
          }
          while ( v43 != v42 );
        }
      }
      v46 = v98;
      v47 = (_QWORD *)(i + v94);
      if ( v98 == v99 )
      {
        sub_278FF40((__int64)&v97, v98, v47);
      }
      else
      {
        if ( v98 )
        {
          *(_QWORD *)v98 = *v47;
          v46 = v98;
        }
        v98 = v46 + 8;
      }
      nullsub_61();
      v117 = &unk_49DA100;
      nullsub_63();
      if ( v102 != (unsigned int *)v104 )
        _libc_free((unsigned __int64)v102);
      if ( v91 )
        sub_B91220((__int64)&v91, v91);
LABEL_58:
      if ( v87 == i )
      {
        v69 = v97;
        v70 = (__int64)&v98[-v97] >> 3;
        if ( (_DWORD)v70 )
        {
          v71 = 0;
          v72 = 8LL * (unsigned int)(v70 - 1);
          while ( 1 )
          {
            sub_B43D60(*(_QWORD **)(v69 + v71));
            v69 = v97;
            if ( v72 == v71 )
              break;
            v71 += 8;
          }
        }
        if ( v69 )
          j_j___libc_free_0(v69);
        goto LABEL_113;
      }
      v12 = v94;
    }
    sub_93FB40((__int64)&v102, 0);
    v23 = v100[0];
    goto LABEL_64;
  }
LABEL_113:
  if ( v94 )
    j_j___libc_free_0(v94);
}
