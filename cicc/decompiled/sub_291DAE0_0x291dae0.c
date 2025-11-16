// Function: sub_291DAE0
// Address: 0x291dae0
//
void __fastcall sub_291DAE0(__int64 *a1, __int64 a2)
{
  unsigned __int64 *v3; // rax
  unsigned __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 *v7; // r12
  unsigned __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 *v10; // rsi
  __int64 v11; // rdx
  unsigned __int64 *v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rcx
  __m128i v15; // xmm1
  __int64 *v16; // r13
  __int64 v17; // r12
  __int64 v18; // rdi
  __int64 *v19; // rax
  _QWORD *v20; // rax
  __int64 v21; // rax
  unsigned __int64 *v22; // r13
  unsigned __int64 v23; // rax
  unsigned __int64 *v24; // rcx
  unsigned __int64 *v25; // rbx
  _QWORD *v26; // r14
  __int64 v27; // r12
  __int64 v28; // rax
  unsigned __int8 v29; // dl
  __int64 v30; // r12
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned __int8 v33; // dl
  __int64 v34; // rsi
  unsigned __int64 *v35; // rax
  unsigned __int64 v36; // r12
  unsigned __int64 *v37; // rsi
  unsigned __int64 *v38; // rdi
  unsigned __int64 *v39; // rax
  unsigned __int64 v40; // r12
  unsigned __int64 *v41; // rsi
  unsigned __int64 *v42; // rdi
  unsigned __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int64 *v45; // rax
  unsigned __int64 v46; // r12
  char v47; // r12
  __int64 v48; // rdx
  __int64 v49; // rbx
  _QWORD *v50; // r13
  char v51; // al
  __int64 v52; // rsi
  __int64 v53; // rbx
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 *v57; // rax
  __int64 v58; // rax
  __int64 v59; // rsi
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // r12
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rsi
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rsi
  __int64 v70; // r12
  __int64 v71; // rax
  unsigned __int64 *v72; // rdx
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // [rsp+0h] [rbp-340h]
  __int64 v76; // [rsp+8h] [rbp-338h]
  __int64 v77; // [rsp+10h] [rbp-330h]
  __int64 v78; // [rsp+38h] [rbp-308h]
  __int64 v79; // [rsp+40h] [rbp-300h]
  unsigned __int8 *v80; // [rsp+48h] [rbp-2F8h]
  __int64 v81; // [rsp+50h] [rbp-2F0h]
  __int64 v82; // [rsp+58h] [rbp-2E8h]
  __int64 v83; // [rsp+60h] [rbp-2E0h]
  unsigned __int64 v84; // [rsp+68h] [rbp-2D8h]
  __int64 v85; // [rsp+78h] [rbp-2C8h]
  __int64 *v86; // [rsp+78h] [rbp-2C8h]
  __int64 v88; // [rsp+88h] [rbp-2B8h]
  __int64 v89; // [rsp+90h] [rbp-2B0h]
  __int64 *v90; // [rsp+98h] [rbp-2A8h]
  __m128i v91; // [rsp+A0h] [rbp-2A0h] BYREF
  __int64 v92; // [rsp+B0h] [rbp-290h]
  __int64 v93; // [rsp+C0h] [rbp-280h] BYREF
  __int64 v94; // [rsp+C8h] [rbp-278h] BYREF
  unsigned __int64 v95; // [rsp+D0h] [rbp-270h] BYREF
  __int64 v96; // [rsp+D8h] [rbp-268h] BYREF
  __m128i v97; // [rsp+E0h] [rbp-260h] BYREF
  __int64 v98; // [rsp+F0h] [rbp-250h]
  _QWORD v99[2]; // [rsp+100h] [rbp-240h] BYREF
  char v100; // [rsp+110h] [rbp-230h]
  __int64 *v101; // [rsp+120h] [rbp-220h] BYREF
  __int64 v102; // [rsp+128h] [rbp-218h]
  _BYTE v103[48]; // [rsp+130h] [rbp-210h] BYREF
  unsigned __int64 *v104[60]; // [rsp+160h] [rbp-1E0h] BYREF

  if ( sub_291DA30(a2) )
    return;
  v80 = sub_291D8E0(a2);
  sub_B12FD0((__int64)v99, a2);
  if ( v100 )
  {
    v3 = (unsigned __int64 *)v99[0];
    v4 = v99[1];
  }
  else
  {
    v104[0] = (unsigned __int64 *)sub_B13000(a2);
    v3 = 0;
    v104[1] = v72;
    if ( (_BYTE)v72 )
      v3 = v104[0];
    v4 = 0;
  }
  v83 = (__int64)v3;
  v84 = v4;
  v93 = 0;
  v101 = (__int64 *)v103;
  v102 = 0x600000000LL;
  v5 = sub_291DAC0(a2);
  if ( !(unsigned __int8)sub_AF4B80(v5, &v93, (__int64)&v101) )
  {
    if ( v101 != (__int64 *)v103 )
      _libc_free((unsigned __int64)v101);
    return;
  }
  v6 = sub_291DAC0(a2);
  v7 = *(unsigned __int64 **)(v6 + 16);
  v8 = *(unsigned __int64 **)(v6 + 24);
  v104[0] = v7;
  if ( v8 == v7 )
  {
LABEL_111:
    v82 = 0;
  }
  else
  {
    while ( *v7 - 4102 > 1 )
    {
      v7 += (unsigned int)sub_AF4160(v104);
      v104[0] = v7;
      if ( v8 == v7 )
        goto LABEL_111;
    }
    v82 = v7[1];
  }
  v9 = a2 + 72;
  v10 = (__int64 *)sub_B43CA0(*a1);
  sub_AE0470((__int64)v104, v10, 0, 0);
  v11 = a1[1];
  v90 = *(__int64 **)v11;
  v78 = *(_QWORD *)v11 + 24LL * *(unsigned int *)(v11 + 8);
  if ( v78 != *(_QWORD *)v11 )
  {
    v89 = a2;
    v12 = (unsigned __int64 *)&v96;
    do
    {
      v13 = v90[1];
      v88 = *v90;
      v14 = v90[2];
      v97 = 0;
      v98 = 0;
      v10 = (__int64 *)*a1;
      if ( (unsigned __int8)sub_AF4D30(a1[2], *a1, v13, v14, (__int64)v80, 8 * v93, v82, v83, v84, (__int64)&v97, &v94) )
      {
        if ( !(_BYTE)v98 )
        {
          sub_B12FD0((__int64)&v91, v89);
          v15 = _mm_loadu_si128(&v91);
          v98 = v92;
          v97 = v15;
LABEL_18:
          v16 = v101;
          v17 = (unsigned int)v102;
          v18 = *a1;
          v79 = v94 - v82;
          if ( v94 - v82 < 0 || v94 == v82 )
          {
            v57 = (__int64 *)sub_BD5C60(v18);
            v81 = sub_B0D000(v57, v16, v17, 0, 1);
          }
          else
          {
            v19 = (__int64 *)sub_BD5C60(v18);
            v20 = (_QWORD *)sub_B0D000(v19, v16, v17, 0, 1);
            v21 = sub_B0DAC0(v20, 0, (v79 + 7) >> 3);
            v79 = 0;
            v81 = v21;
          }
          v22 = &v95;
          sub_AE74C0(&v95, v88);
          v23 = v95;
          if ( (v95 & 4) == 0 )
          {
            v24 = v12;
            if ( (v95 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
              goto LABEL_23;
LABEL_37:
            if ( v23 )
            {
              if ( (v23 & 4) != 0 )
              {
                v35 = (unsigned __int64 *)(v23 & 0xFFFFFFFFFFFFFFF8LL);
                v36 = (unsigned __int64)v35;
                if ( v35 )
                {
                  if ( (unsigned __int64 *)*v35 != v35 + 2 )
                    _libc_free(*v35);
                  j_j___libc_free_0(v36);
                }
              }
            }
            sub_AE7690(v12, v88);
            if ( (v96 & 4) != 0 )
            {
              v38 = *(unsigned __int64 **)(v96 & 0xFFFFFFFFFFFFFFF8LL);
              v37 = &v38[*(unsigned int *)((v96 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
            }
            else
            {
              v37 = (unsigned __int64 *)&v97;
              v38 = v12;
              if ( (v96 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                v37 = v12;
            }
            sub_29141B0((_QWORD **)v38, (_QWORD **)v37, v89);
            if ( v96 )
            {
              if ( (v96 & 4) != 0 )
              {
                v39 = (unsigned __int64 *)(v96 & 0xFFFFFFFFFFFFFFF8LL);
                v40 = v96 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v96 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                {
                  if ( (unsigned __int64 *)*v39 != v39 + 2 )
                    _libc_free(*v39);
                  j_j___libc_free_0(v40);
                }
              }
            }
            sub_AE7870(v12, v88);
            if ( (v96 & 4) != 0 )
            {
              v42 = *(unsigned __int64 **)(v96 & 0xFFFFFFFFFFFFFFF8LL);
              v41 = &v42[*(unsigned int *)((v96 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
            }
            else
            {
              v41 = (unsigned __int64 *)&v97;
              v42 = v12;
              if ( (v96 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
                v41 = v12;
            }
            sub_29141B0((_QWORD **)v42, (_QWORD **)v41, v89);
            if ( v96 )
            {
              if ( (v96 & 4) != 0 )
              {
                v45 = (unsigned __int64 *)(v96 & 0xFFFFFFFFFFFFFFF8LL);
                v46 = v96 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v96 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
                {
                  if ( (unsigned __int64 *)*v45 != v45 + 2 )
                    _libc_free(*v45);
                  j_j___libc_free_0(v46);
                }
              }
            }
            v47 = v98;
            v48 = v97.m128i_i64[1];
            v10 = (__int64 *)v97.m128i_i64[0];
            v49 = *a1;
            v50 = (_QWORD *)v81;
            if ( *(_BYTE *)(v89 + 64) == 2 )
            {
              v76 = v97.m128i_i64[1];
              v86 = (__int64 *)v97.m128i_i64[0];
              v64 = sub_B11F60(v89 + 80);
              v48 = v76;
              v10 = v86;
              v50 = (_QWORD *)v64;
            }
            if ( v47 )
              v50 = (_QWORD *)sub_2916270(v50, (unsigned __int64)v10, v48, v79, v43, v44);
            if ( v50 )
            {
              v51 = *(_BYTE *)(v89 + 64);
              if ( v51 )
              {
                if ( v51 == 1 )
                {
                  v65 = *(_QWORD *)(v89 + 24);
                  v96 = v65;
                  if ( v65 )
                    sub_B96E90((__int64)v12, v65, 1);
                  v66 = sub_B10CD0((__int64)v12);
                  v67 = sub_B12000(v9);
                  v68 = sub_B12800(v88, v67, (__int64)v50, v66);
                  v69 = v96;
                  v70 = v68;
                  if ( v96 )
                    sub_B91220((__int64)v12, v96);
                  if ( !sub_AF4730((__int64)v50) )
                    sub_B14010(v70, v69);
                  v71 = v75;
                  v10 = (__int64 *)v70;
                  LOWORD(v71) = 0;
                  v75 = v71;
                  sub_AA8770(*(_QWORD *)(v49 + 40), v70, v49 + 24, 0);
                }
                else
                {
                  if ( (*(_BYTE *)(v88 + 7) & 0x20) == 0 || !sub_B91C10(v88, 38) )
                  {
                    v73 = sub_BD5C60(v88);
                    v74 = sub_AF40E0(v73, 1u);
                    sub_B99FD0(v88, 0x26u, v74);
                  }
                  v52 = *(_QWORD *)(v89 + 24);
                  v96 = v52;
                  if ( v52 )
                    sub_B96E90((__int64)v12, v52, 1);
                  v53 = sub_B10CD0((__int64)v12);
                  v54 = sub_B12000(v9);
                  v55 = sub_B12A50(v89, 0);
                  sub_B12940(v88, v55, v54, (__int64)v50, v88, v81, v53);
                  v10 = (__int64 *)v96;
                  if ( v96 )
                    sub_B91220((__int64)v12, v96);
                }
              }
              else
              {
                v59 = *(_QWORD *)(v89 + 24);
                v96 = v59;
                if ( v59 )
                  sub_B96E90((__int64)v12, v59, 1);
                v60 = sub_B10CD0((__int64)v12);
                v61 = sub_B12000(v9);
                v62 = sub_B12860(v88, v61, (__int64)v50, v60);
                if ( v96 )
                  sub_B91220((__int64)v12, v96);
                v63 = v77;
                v10 = (__int64 *)v62;
                LOWORD(v63) = 0;
                v77 = v63;
                sub_AA8770(*(_QWORD *)(v49 + 40), v62, v49 + 24, 0);
              }
            }
            goto LABEL_74;
          }
          v22 = *(unsigned __int64 **)(v95 & 0xFFFFFFFFFFFFFFF8LL);
          v24 = &v22[*(unsigned int *)((v95 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
          if ( v22 == v24 )
            goto LABEL_37;
LABEL_23:
          v85 = (__int64)v12;
          v25 = v24;
          while ( 2 )
          {
            while ( 2 )
            {
              v26 = (_QWORD *)*v22;
              v27 = *(_QWORD *)(*(_QWORD *)(*v22 + 32 * (1LL - (*(_DWORD *)(*v22 + 4) & 0x7FFFFFF))) + 24LL);
              if ( v27 != sub_B12000(v9) )
                goto LABEL_24;
              v28 = sub_B10CD0((__int64)(v26 + 6));
              v29 = *(_BYTE *)(v28 - 16);
              if ( (v29 & 2) != 0 )
              {
                if ( *(_DWORD *)(v28 - 24) != 2 )
                  goto LABEL_28;
                v58 = *(_QWORD *)(v28 - 32);
LABEL_91:
                v30 = *(_QWORD *)(v58 + 8);
              }
              else
              {
                if ( ((*(_WORD *)(v28 - 16) >> 6) & 0xF) == 2 )
                {
                  v58 = v28 - 16 - 8LL * ((v29 >> 2) & 0xF);
                  goto LABEL_91;
                }
LABEL_28:
                v30 = 0;
              }
              v31 = *(_QWORD *)(v89 + 24);
              v96 = v31;
              if ( v31 )
                sub_B96E90(v85, v31, 1);
              v32 = sub_B10CD0(v85);
              v33 = *(_BYTE *)(v32 - 16);
              if ( (v33 & 2) != 0 )
              {
                if ( *(_DWORD *)(v32 - 24) != 2 )
                {
LABEL_33:
                  v34 = v96;
                  if ( !v30 )
                    goto LABEL_79;
LABEL_34:
                  if ( !v34 )
                    goto LABEL_24;
                  ++v22;
                  sub_B91220(v85, v34);
                  if ( v22 == v25 )
                  {
LABEL_36:
                    v12 = (unsigned __int64 *)v85;
                    v23 = v95;
                    goto LABEL_37;
                  }
                  continue;
                }
                v56 = *(_QWORD *)(v32 - 32);
              }
              else
              {
                if ( ((*(_WORD *)(v32 - 16) >> 6) & 0xF) != 2 )
                  goto LABEL_33;
                v56 = v32 - 16 - 8LL * ((v33 >> 2) & 0xF);
              }
              break;
            }
            v34 = v96;
            if ( *(_QWORD *)(v56 + 8) == v30 )
            {
LABEL_79:
              if ( v34 )
                sub_B91220(v85, v34);
              sub_B43D60(v26);
LABEL_24:
              if ( ++v22 == v25 )
                goto LABEL_36;
              continue;
            }
            goto LABEL_34;
          }
        }
        if ( v97.m128i_i64[0] )
          goto LABEL_18;
      }
LABEL_74:
      v90 += 3;
    }
    while ( (__int64 *)v78 != v90 );
  }
  sub_AE9130((__int64)v104, (__int64)v10);
  if ( v101 != (__int64 *)v103 )
    _libc_free((unsigned __int64)v101);
}
