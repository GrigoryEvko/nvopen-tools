// Function: sub_2C32020
// Address: 0x2c32020
//
__int64 __fastcall sub_2C32020(__int64 a1)
{
  _QWORD *v1; // r12
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r9
  __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  _BYTE *v32; // rsi
  __int64 v33; // r8
  __int64 v34; // rax
  __int64 v35; // rax
  char v36; // di
  char v37; // al
  char *v38; // rax
  char *v39; // rdx
  __int64 *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // r11
  __int64 v47; // rax
  __int64 v48; // r13
  _QWORD *v49; // r15
  _QWORD *v50; // r12
  __int64 v51; // r15
  unsigned __int64 *v52; // rax
  __int64 v53; // rax
  __int64 v54; // r15
  unsigned __int64 v55; // r12
  __int64 *v56; // r13
  __int64 v57; // rsi
  __int64 v58; // rdx
  unsigned __int64 v59; // rbx
  _QWORD **v60; // rdi
  __int64 v61; // rax
  _QWORD *v62; // rbx
  _QWORD *v63; // r13
  __int64 *v64; // rdi
  __int64 *v65; // rbx
  __int64 v66; // r15
  _QWORD *v67; // rdi
  __int64 v68; // rsi
  _QWORD *v69; // rax
  int v70; // r9d
  _QWORD *v71; // rdi
  __int64 v72; // rsi
  _QWORD *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  int v77; // r9d
  __int64 v78; // r9
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  _QWORD *v83; // rdi
  __int64 v84; // rsi
  _QWORD *v85; // rax
  int v86; // r8d
  _QWORD *v87; // rdi
  __int64 v88; // rsi
  _QWORD *v89; // rax
  int v90; // r8d
  __int64 v91; // [rsp+8h] [rbp-728h]
  _QWORD *v92; // [rsp+8h] [rbp-728h]
  _BYTE *v93; // [rsp+10h] [rbp-720h]
  __int64 v94; // [rsp+10h] [rbp-720h]
  __int64 v95; // [rsp+18h] [rbp-718h]
  __int64 v96; // [rsp+18h] [rbp-718h]
  _QWORD *v97; // [rsp+20h] [rbp-710h]
  __int64 v98; // [rsp+20h] [rbp-710h]
  __int64 v99; // [rsp+28h] [rbp-708h]
  _QWORD *v100; // [rsp+30h] [rbp-700h]
  __int64 v101; // [rsp+38h] [rbp-6F8h]
  unsigned __int64 v102; // [rsp+38h] [rbp-6F8h]
  __int64 *v103; // [rsp+38h] [rbp-6F8h]
  __int64 v104; // [rsp+40h] [rbp-6F0h] BYREF
  char *v105; // [rsp+48h] [rbp-6E8h]
  __int64 v106; // [rsp+50h] [rbp-6E0h]
  int v107; // [rsp+58h] [rbp-6D8h]
  char v108; // [rsp+5Ch] [rbp-6D4h]
  char v109; // [rsp+60h] [rbp-6D0h] BYREF
  _BYTE *v110; // [rsp+80h] [rbp-6B0h] BYREF
  __int64 v111; // [rsp+88h] [rbp-6A8h]
  _BYTE v112[64]; // [rsp+90h] [rbp-6A0h] BYREF
  _QWORD v113[15]; // [rsp+D0h] [rbp-660h] BYREF
  char v114[120]; // [rsp+148h] [rbp-5E8h] BYREF
  _QWORD v115[12]; // [rsp+1C0h] [rbp-570h] BYREF
  __int64 v116; // [rsp+220h] [rbp-510h]
  unsigned __int64 v117; // [rsp+228h] [rbp-508h]
  __int16 v118; // [rsp+238h] [rbp-4F8h]
  _QWORD v119[12]; // [rsp+240h] [rbp-4F0h] BYREF
  _BYTE *v120; // [rsp+2A0h] [rbp-490h]
  __int64 v121; // [rsp+2A8h] [rbp-488h]
  __int16 v122; // [rsp+2B8h] [rbp-478h]
  __int16 v123; // [rsp+2C8h] [rbp-468h]
  _QWORD v124[12]; // [rsp+2D0h] [rbp-460h] BYREF
  _BYTE *v125; // [rsp+330h] [rbp-400h]
  __int64 v126; // [rsp+338h] [rbp-3F8h]
  __int16 v127; // [rsp+348h] [rbp-3E8h]
  _QWORD v128[15]; // [rsp+350h] [rbp-3E0h] BYREF
  __int16 v129; // [rsp+3C8h] [rbp-368h]
  __int16 v130[64]; // [rsp+3D8h] [rbp-358h] BYREF
  char v131[136]; // [rsp+458h] [rbp-2D8h] BYREF
  __int64 v132[15]; // [rsp+4E0h] [rbp-250h] BYREF
  __int16 v133; // [rsp+558h] [rbp-1D8h]
  _BYTE v134[120]; // [rsp+560h] [rbp-1D0h] BYREF
  __int16 v135; // [rsp+5D8h] [rbp-158h]
  __int16 v136; // [rsp+5E8h] [rbp-148h]
  _BYTE v137[120]; // [rsp+5F0h] [rbp-140h] BYREF
  __int16 v138; // [rsp+668h] [rbp-C8h]
  _BYTE v139[120]; // [rsp+670h] [rbp-C0h] BYREF
  __int16 v140; // [rsp+6E8h] [rbp-48h]
  __int16 v141; // [rsp+6F8h] [rbp-38h]

  v1 = v115;
  v105 = &v109;
  v110 = v112;
  v111 = 0x800000000LL;
  v104 = 0;
  v106 = 4;
  v107 = 0;
  v108 = 1;
  sub_2C2F4B0(v113, a1);
  sub_2C2B410((__int64)v115, (__int64)v113, v2, v3, v4, v5);
  sub_2C31AD0((__int64)v124, (__int64)v115, v6, v7, v8, v9);
  sub_2C2B5D0((__int64)v132, v124, v10, v11, v12, v13);
  sub_2AB1B50((__int64)v131);
  sub_2AB1B50((__int64)v130);
  sub_2AB1B50((__int64)v128);
  sub_2AB1B50((__int64)v124);
  sub_2AB1B50((__int64)v119);
  sub_2AB1B50((__int64)v115);
  sub_2AB1B50((__int64)v114);
  sub_2AB1B50((__int64)v113);
  sub_2ABCC20(v115, (__int64)v132, v14, v15, v16, v17);
  v118 = v133;
  sub_2ABCC20(v119, (__int64)v134, v18, (__int64)v134, v19, v20);
  v122 = v135;
  v123 = v136;
  sub_2ABCC20(v124, (__int64)v137, v21, v22, v23, v24);
  v127 = v138;
  sub_2ABCC20(v128, (__int64)v139, v25, v26, v27, v28);
  v30 = v116;
  v31 = v117;
  v129 = v140;
  v130[0] = v141;
LABEL_2:
  v32 = v125;
  if ( v31 - v30 != v126 - (_QWORD)v125 )
    goto LABEL_3;
  if ( v31 != v30 )
  {
    while ( *(_QWORD *)v30 == *(_QWORD *)v32 )
    {
      v37 = *(_BYTE *)(v30 + 24);
      if ( v37 != v32[24]
        || v37 && (*(_QWORD *)(v30 + 8) != *((_QWORD *)v32 + 1) || *(_QWORD *)(v30 + 16) != *((_QWORD *)v32 + 2)) )
      {
        break;
      }
      v30 += 32;
      v32 += 32;
      if ( v30 == v31 )
        goto LABEL_27;
    }
LABEL_3:
    v33 = *(_QWORD *)(v31 - 32);
    if ( *(_BYTE *)(v33 + 128) )
    {
      if ( *(_DWORD *)(v33 + 88) == 1 )
      {
        v34 = **(_QWORD **)(v33 + 80);
        if ( v34 )
        {
          v31 = (unsigned int)*(unsigned __int8 *)(v34 + 8) - 1;
          if ( (unsigned int)v31 <= 1 )
          {
            v30 = v34 + 112;
            v31 = *(_QWORD *)(v34 + 112) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v34 + 112 == v31 && *(_DWORD *)(v34 + 88) == 1 )
            {
              v41 = *(__int64 **)(v34 + 80);
              v29 = *v41;
              if ( *v41 )
              {
                if ( !*(_BYTE *)(v29 + 8) )
                {
                  v95 = *v41;
                  if ( *(_BYTE *)(v29 + 128) )
                  {
                    v91 = v33;
                    v94 = sub_2C291F0(v33);
                    v42 = sub_2C291F0(v95);
                    v31 = v94;
                    if ( v94 )
                    {
                      if ( v94 == v42 )
                      {
                        v43 = (unsigned int)v111;
                        v30 = HIDWORD(v111);
                        v33 = v91;
                        v44 = (unsigned int)v111 + 1LL;
                        if ( v44 > HIDWORD(v111) )
                        {
                          v32 = v112;
                          sub_C8D5F0((__int64)&v110, v112, v44, 8u, v91, v29);
                          v43 = (unsigned int)v111;
                          v33 = v91;
                        }
                        v31 = (unsigned __int64)v110;
                        *(_QWORD *)&v110[8 * v43] = v33;
                        LODWORD(v111) = v111 + 1;
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    while ( 1 )
    {
      sub_2AD7320((__int64)v115, (__int64)v32, v31, v30, v33, v29);
      v31 = v117;
      v30 = v116;
      v32 = v120;
      if ( v117 - v116 == v121 - (_QWORD)v120 )
      {
        if ( v116 == v117 )
          goto LABEL_2;
        v35 = v116;
        while ( *(_QWORD *)v35 == *(_QWORD *)v32 )
        {
          v36 = *(_BYTE *)(v35 + 24);
          if ( v36 != v32[24]
            || v36 && (*(_QWORD *)(v35 + 8) != *((_QWORD *)v32 + 1) || *(_QWORD *)(v35 + 16) != *((_QWORD *)v32 + 2)) )
          {
            break;
          }
          v35 += 32;
          v32 += 32;
          if ( v117 == v35 )
            goto LABEL_2;
        }
      }
      if ( !*(_BYTE *)(*(_QWORD *)(v117 - 32) + 8LL) )
        goto LABEL_2;
    }
  }
LABEL_27:
  sub_2AB1B50((__int64)v128);
  sub_2AB1B50((__int64)v124);
  sub_2AB1B50((__int64)v119);
  sub_2AB1B50((__int64)v115);
  sub_2AB1B50((__int64)v139);
  sub_2AB1B50((__int64)v137);
  sub_2AB1B50((__int64)v134);
  sub_2AB1B50((__int64)v132);
  v93 = &v110[8 * (unsigned int)v111];
  if ( v110 != v93 )
  {
    v100 = v110;
    while ( 1 )
    {
      v1 = (_QWORD *)*v100;
      if ( v108 )
      {
        v38 = v105;
        v39 = &v105[8 * HIDWORD(v106)];
        if ( v105 == v39 )
          goto LABEL_51;
        while ( v1 != *(_QWORD **)v38 )
        {
          v38 += 8;
          if ( v39 == v38 )
            goto LABEL_51;
        }
      }
      else if ( !sub_C8CA60((__int64)&v104, (__int64)v1) )
      {
LABEL_51:
        if ( *((_DWORD *)v1 + 22) != 1 )
          BUG();
        v45 = (__int64 *)v1[10];
        v99 = *v45;
        if ( *(_DWORD *)(*v45 + 88) != 1 )
          BUG();
        v101 = sub_2C25A30(v1[14]);
        v47 = sub_2C25A30(*(_QWORD *)(**(_QWORD **)(v46 + 80) + 112LL));
        v48 = v47;
        if ( v101 && v47 )
        {
          v49 = (_QWORD *)(*(_QWORD *)(v101 + 112) & 0xFFFFFFFFFFFFFFF8LL);
          if ( (_QWORD *)(v101 + 112) != v49 )
          {
            v97 = v1;
            do
            {
              v50 = v49;
              v51 = *v49;
              v52 = (unsigned __int64 *)sub_2BF05A0(v48);
              v49 = (_QWORD *)(v51 & 0xFFFFFFFFFFFFFFF8LL);
              sub_2C19EE0(v50 - 3, v48, v52);
            }
            while ( (_QWORD *)(v101 + 112) != v49 );
            v1 = v97;
          }
          if ( *(_DWORD *)(v101 + 88) == 1 )
            v53 = **(_QWORD **)(v101 + 80);
          else
            v53 = 0;
          v54 = 0;
          if ( *(_DWORD *)(v48 + 88) == 1 )
            v54 = **(_QWORD **)(v48 + 80);
          v98 = v53 + 112;
          v102 = *(_QWORD *)(v53 + 112) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v53 + 112 != v102 )
          {
            v92 = v1;
            v96 = v48;
            while ( 1 )
            {
              v55 = v102;
              v56 = (__int64 *)(v102 - 24);
              v57 = **(_QWORD **)(v102 + 24);
              v58 = *(_QWORD *)(v102 - 8);
              v59 = *(_QWORD *)v102 & 0xFFFFFFFFFFFFFFF8LL;
              v102 = v59;
              v60 = (_QWORD **)(v58 & 0xFFFFFFFFFFFFFFF8LL);
              if ( (v58 & 4) != 0 )
                v60 = (_QWORD **)**v60;
              v132[0] = v96;
              sub_2BF1090(
                (__int64)v60,
                v57,
                (unsigned __int8 (__fastcall *)(__int64, __int64, _QWORD))sub_2C250B0,
                (__int64)v132);
              v61 = *(_QWORD *)(v55 - 8);
              if ( (v61 & 4) != 0 )
              {
                if ( !*(_DWORD *)(**(_QWORD **)(v61 & 0xFFFFFFFFFFFFFFF8LL) + 24LL) )
                  goto LABEL_71;
LABEL_66:
                sub_2C19EE0(v56, v54, *(unsigned __int64 **)(v54 + 120));
                if ( v98 == v59 )
                  goto LABEL_72;
              }
              else
              {
                if ( *(_DWORD *)((v61 & 0xFFFFFFFFFFFFFFF8LL) + 24) )
                  goto LABEL_66;
LABEL_71:
                sub_2C19E60(v56);
                if ( v98 == v59 )
                {
LABEL_72:
                  v1 = v92;
                  break;
                }
              }
            }
          }
          v62 = (_QWORD *)(sub_2BF04D0((__int64)v1) + 112);
          v63 = (_QWORD *)(*v62 & 0xFFFFFFFFFFFFFFF8LL);
          while ( v62 != v63 )
          {
            v64 = v63 - 3;
            v63 = (_QWORD *)(*v63 & 0xFFFFFFFFFFFFFFF8LL);
            sub_2C19E60(v64);
          }
          v65 = (__int64 *)v1[7];
          v103 = &v65[*((unsigned int *)v1 + 16)];
          while ( v103 != v65 )
          {
            v66 = *v65;
            v132[0] = (__int64)v1;
            v67 = *(_QWORD **)(v66 + 80);
            v68 = (__int64)&v67[*(unsigned int *)(v66 + 88)];
            v69 = sub_2C25750(v67, v68, v132);
            if ( v69 + 1 != (_QWORD *)v68 )
            {
              memmove(v69, v69 + 1, v68 - (_QWORD)(v69 + 1));
              v70 = *(_DWORD *)(v66 + 88);
            }
            *(_DWORD *)(v66 + 88) = v70 - 1;
            v132[0] = v66;
            v71 = (_QWORD *)v1[7];
            v72 = (__int64)&v71[*((unsigned int *)v1 + 16)];
            v73 = sub_2C25750(v71, v72, v132);
            if ( v73 + 1 != (_QWORD *)v72 )
            {
              memmove(v73, v73 + 1, v72 - (_QWORD)(v73 + 1));
              v77 = *((_DWORD *)v1 + 16);
            }
            v78 = (unsigned int)(v77 - 1);
            ++v65;
            *((_DWORD *)v1 + 16) = v78;
            sub_2AB9570(v66 + 80, v99, v74, v75, v76, v78);
            sub_2AB9570(v99 + 56, v66, v79, v80, v81, v82);
          }
          v132[0] = v99;
          v83 = (_QWORD *)v1[10];
          v84 = (__int64)&v83[*((unsigned int *)v1 + 22)];
          v85 = sub_2C25750(v83, v84, v132);
          if ( v85 + 1 != (_QWORD *)v84 )
          {
            memmove(v85, v85 + 1, v84 - (_QWORD)(v85 + 1));
            v86 = *((_DWORD *)v1 + 22);
          }
          *((_DWORD *)v1 + 22) = v86 - 1;
          v132[0] = (__int64)v1;
          v87 = *(_QWORD **)(v99 + 56);
          v88 = (__int64)&v87[*(unsigned int *)(v99 + 64)];
          v89 = sub_2C25750(v87, v88, v132);
          if ( v89 + 1 != (_QWORD *)v88 )
          {
            memmove(v89, v89 + 1, v88 - (_QWORD)(v89 + 1));
            v90 = *(_DWORD *)(v99 + 64);
          }
          *(_DWORD *)(v99 + 64) = v90 - 1;
          sub_AE6EC0((__int64)&v104, (__int64)v1);
        }
      }
      if ( v93 == (_BYTE *)++v100 )
      {
        v93 = v110;
        break;
      }
    }
  }
  LOBYTE(v1) = HIDWORD(v106) != v107;
  if ( v93 != v112 )
    _libc_free((unsigned __int64)v93);
  if ( !v108 )
    _libc_free((unsigned __int64)v105);
  return (unsigned int)v1;
}
