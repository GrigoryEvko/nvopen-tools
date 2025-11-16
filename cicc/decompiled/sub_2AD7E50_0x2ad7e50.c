// Function: sub_2AD7E50
// Address: 0x2ad7e50
//
void __fastcall sub_2AD7E50(__int64 a1, __int64 a2)
{
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
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rax
  unsigned __int64 v33; // rsi
  __int64 v34; // rbx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 v41; // r12
  __int64 v42; // rax
  unsigned int v43; // r13d
  __int64 v44; // r9
  int v45; // esi
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 v48; // r10
  __int64 v49; // r12
  __int64 v50; // rdx
  int v51; // eax
  int v52; // eax
  __int64 *v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // r8
  __int64 v56; // r12
  char v57; // cl
  unsigned int v58; // esi
  unsigned int v59; // eax
  __int64 *v60; // rdi
  int v61; // edx
  unsigned int v62; // r9d
  int v63; // r11d
  unsigned __int64 v64; // rax
  __int64 v65; // rcx
  int v66; // edx
  unsigned int v67; // eax
  __int64 v68; // rsi
  __int64 v69; // rcx
  int v70; // edx
  unsigned int v71; // eax
  __int64 v72; // rsi
  int v73; // r11d
  __int64 *v74; // r9
  int v75; // edx
  int v76; // edx
  __int64 v77; // rcx
  char v78; // di
  char v79; // cl
  unsigned __int64 v80; // r13
  int v81; // r11d
  unsigned __int64 v82; // rdx
  __int64 v83; // [rsp+48h] [rbp-6B8h]
  __int64 v84; // [rsp+58h] [rbp-6A8h]
  __int64 v85; // [rsp+60h] [rbp-6A0h]
  __int64 v86; // [rsp+68h] [rbp-698h]
  __int64 v88; // [rsp+78h] [rbp-688h]
  unsigned __int64 v89; // [rsp+88h] [rbp-678h]
  __int64 v90; // [rsp+88h] [rbp-678h]
  __int64 v91; // [rsp+98h] [rbp-668h] BYREF
  _QWORD v92[15]; // [rsp+A0h] [rbp-660h] BYREF
  char v93[120]; // [rsp+118h] [rbp-5E8h] BYREF
  _QWORD v94[12]; // [rsp+190h] [rbp-570h] BYREF
  __int64 v95; // [rsp+1F0h] [rbp-510h]
  __int64 v96; // [rsp+1F8h] [rbp-508h]
  __int16 v97; // [rsp+208h] [rbp-4F8h]
  _QWORD v98[12]; // [rsp+210h] [rbp-4F0h] BYREF
  unsigned __int64 v99; // [rsp+270h] [rbp-490h]
  __int64 v100; // [rsp+278h] [rbp-488h]
  __int16 v101; // [rsp+288h] [rbp-478h]
  __int16 v102; // [rsp+298h] [rbp-468h]
  _QWORD v103[12]; // [rsp+2A0h] [rbp-460h] BYREF
  unsigned __int64 v104; // [rsp+300h] [rbp-400h]
  __int64 v105; // [rsp+308h] [rbp-3F8h]
  __int16 v106; // [rsp+318h] [rbp-3E8h]
  _QWORD v107[15]; // [rsp+320h] [rbp-3E0h] BYREF
  __int16 v108; // [rsp+398h] [rbp-368h]
  __int16 v109[64]; // [rsp+3A8h] [rbp-358h] BYREF
  char v110[136]; // [rsp+428h] [rbp-2D8h] BYREF
  _BYTE v111[120]; // [rsp+4B0h] [rbp-250h] BYREF
  __int16 v112; // [rsp+528h] [rbp-1D8h]
  _BYTE v113[120]; // [rsp+530h] [rbp-1D0h] BYREF
  __int16 v114; // [rsp+5A8h] [rbp-158h]
  __int16 v115; // [rsp+5B8h] [rbp-148h]
  _BYTE v116[120]; // [rsp+5C0h] [rbp-140h] BYREF
  __int16 v117; // [rsp+638h] [rbp-C8h]
  _BYTE v118[120]; // [rsp+640h] [rbp-C0h] BYREF
  __int16 v119; // [rsp+6B8h] [rbp-48h]
  __int16 v120; // [rsp+6C8h] [rbp-38h]

  sub_2AC67F0(v92, **(_QWORD **)(a1 + 464));
  sub_2AB1B90((__int64)v94, (__int64)v92, v2, v3, v4, v5);
  sub_2AD74E0((__int64)v103, (__int64)v94, v6, v7, v8, v9);
  sub_2ABCD20((__int64)v111, v103, v10, v11, v12, v13);
  sub_2AB1B50((__int64)v110);
  sub_2AB1B50((__int64)v109);
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v103);
  sub_2AB1B50((__int64)v98);
  sub_2AB1B50((__int64)v94);
  sub_2ABCC20(v94, (__int64)v111, v14, v15, v16, v17);
  v97 = v112;
  sub_2ABCC20(v98, (__int64)v113, v18, v19, v20, v21);
  v101 = v114;
  v102 = v115;
  sub_2ABCC20(v103, (__int64)v116, v23, v24, v25, v26);
  v106 = v117;
  sub_2ABCC20(v107, (__int64)v118, v27, v28, v29, v30);
  v31 = v95;
  v108 = v119;
  v109[0] = v120;
  v32 = v96;
LABEL_2:
  v33 = v104;
  if ( v32 - v31 != v105 - v104 )
  {
LABEL_3:
    v34 = *(_QWORD *)(v32 - 32);
    v86 = sub_2BF05A0(v34);
    v84 = a1 + 96;
    if ( *(_QWORD *)(v34 + 120) == v86 )
      goto LABEL_34;
    v88 = *(_QWORD *)(v34 + 120);
    while ( 1 )
    {
      if ( !v88 )
        BUG();
      if ( *(_BYTE *)(v88 - 16) != 27 )
        goto LABEL_33;
      v38 = sub_2BFB640(a2, v88 + 72, 0);
      v39 = *(_QWORD *)(v38 + 40);
      *(_WORD *)(a1 + 160) = 0;
      *(_QWORD *)(a1 + 144) = v39;
      *(_QWORD *)(a1 + 152) = v38 + 24;
      v40 = *(_QWORD *)sub_B46C60(v38);
      v91 = v40;
      if ( v40 && (sub_B96E90((__int64)&v91, v40, 1), (v41 = v91) != 0) )
      {
        v33 = *(unsigned int *)(a1 + 104);
        v42 = *(_QWORD *)(a1 + 96);
        v35 = v33;
        v31 = v42 + 16 * v33;
        if ( v42 != v31 )
        {
          while ( *(_DWORD *)v42 )
          {
            v42 += 16;
            if ( v31 == v42 )
              goto LABEL_53;
          }
          *(_QWORD *)(v42 + 8) = v91;
LABEL_14:
          v33 = v41;
          sub_B91220((__int64)&v91, v41);
          goto LABEL_15;
        }
LABEL_53:
        v64 = *(unsigned int *)(a1 + 108);
        if ( v33 >= v64 )
        {
          ++v33;
          v80 = v83 & 0xFFFFFFFF00000000LL;
          v83 &= 0xFFFFFFFF00000000LL;
          if ( v64 < v33 )
          {
            v82 = v33;
            v33 = a1 + 112;
            sub_C8D5F0(v84, (const void *)(a1 + 112), v82, 0x10u, a1 + 112, v37);
            v31 = *(_QWORD *)(a1 + 96) + 16LL * *(unsigned int *)(a1 + 104);
          }
          *(_QWORD *)v31 = v80;
          *(_QWORD *)(v31 + 8) = v41;
          v41 = v91;
          ++*(_DWORD *)(a1 + 104);
        }
        else
        {
          if ( v31 )
          {
            *(_DWORD *)v31 = 0;
            *(_QWORD *)(v31 + 8) = v41;
            v41 = v91;
            LODWORD(v35) = *(_DWORD *)(a1 + 104);
          }
          v35 = (unsigned int)(v35 + 1);
          *(_DWORD *)(a1 + 104) = v35;
        }
      }
      else
      {
        v33 = 0;
        sub_93FB40(v84, 0);
        v41 = v91;
      }
      if ( v41 )
        goto LABEL_14;
LABEL_15:
      if ( *(_DWORD *)(v88 + 32) )
      {
        v43 = 0;
        v85 = a2 + 120;
        while ( 1 )
        {
          v89 = *(_QWORD *)(*(_QWORD *)(v88 + 24) + 8LL * v43);
          v54 = sub_2C1B6B0(v88 - 24, v43);
          v55 = v89;
          v56 = v54;
          v57 = *(_BYTE *)(a2 + 128) & 1;
          if ( v57 )
          {
            v44 = a2 + 136;
            v45 = 3;
          }
          else
          {
            v58 = *(_DWORD *)(a2 + 144);
            v44 = *(_QWORD *)(a2 + 136);
            if ( !v58 )
            {
              v59 = *(_DWORD *)(a2 + 128);
              ++*(_QWORD *)(a2 + 120);
              v60 = 0;
              v61 = (v59 >> 1) + 1;
              goto LABEL_38;
            }
            v45 = v58 - 1;
          }
          v46 = v45 & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
          v47 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v47;
          if ( v56 != *v47 )
            break;
LABEL_19:
          v49 = v47[1];
LABEL_20:
          v33 = v55;
          v50 = sub_2BFB640(a2, v55, 0);
          v51 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
          if ( v51 == *(_DWORD *)(v38 + 72) )
          {
            v90 = v50;
            sub_B48D90(v38);
            v50 = v90;
            v51 = *(_DWORD *)(v38 + 4) & 0x7FFFFFF;
          }
          v52 = (v51 + 1) & 0x7FFFFFF;
          v35 = v52 | *(_DWORD *)(v38 + 4) & 0xF8000000;
          v53 = (__int64 *)(*(_QWORD *)(v38 - 8) + 32LL * (unsigned int)(v52 - 1));
          *(_DWORD *)(v38 + 4) = v35;
          if ( *v53 )
          {
            v33 = v53[2];
            v35 = v53[1];
            *(_QWORD *)v33 = v35;
            if ( v35 )
            {
              v33 = v53[2];
              *(_QWORD *)(v35 + 16) = v33;
            }
          }
          *v53 = v50;
          if ( v50 )
          {
            v35 = *(_QWORD *)(v50 + 16);
            v33 = v50 + 16;
            v53[1] = v35;
            if ( v35 )
              *(_QWORD *)(v35 + 16) = v53 + 1;
            v53[2] = v33;
            *(_QWORD *)(v50 + 16) = v53;
          }
          ++v43;
          v31 = (*(_DWORD *)(v38 + 4) & 0x7FFFFFFu) - 1;
          *(_QWORD *)(*(_QWORD *)(v38 - 8) + 32LL * *(unsigned int *)(v38 + 72) + 8 * v31) = v49;
          if ( v43 >= *(_DWORD *)(v88 + 32) )
            goto LABEL_33;
        }
        v63 = 1;
        v60 = 0;
        while ( v48 != -4096 )
        {
          if ( !v60 && v48 == -8192 )
            v60 = v47;
          v46 = v45 & (v63 + v46);
          v47 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v47;
          if ( v56 == *v47 )
            goto LABEL_19;
          ++v63;
        }
        v62 = 12;
        v58 = 4;
        if ( !v60 )
          v60 = v47;
        v59 = *(_DWORD *)(a2 + 128);
        ++*(_QWORD *)(a2 + 120);
        v61 = (v59 >> 1) + 1;
        if ( !v57 )
        {
          v58 = *(_DWORD *)(a2 + 144);
LABEL_38:
          v62 = 3 * v58;
        }
        if ( 4 * v61 >= v62 )
        {
          sub_2ACA3E0(v85, 2 * v58);
          v55 = v89;
          if ( (*(_BYTE *)(a2 + 128) & 1) != 0 )
          {
            v65 = a2 + 136;
            v66 = 3;
          }
          else
          {
            v75 = *(_DWORD *)(a2 + 144);
            v65 = *(_QWORD *)(a2 + 136);
            if ( !v75 )
              goto LABEL_111;
            v66 = v75 - 1;
          }
          v67 = v66 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v60 = (__int64 *)(v65 + 16LL * v67);
          v68 = *v60;
          if ( v56 != *v60 )
          {
            v81 = 1;
            v74 = 0;
            while ( v68 != -4096 )
            {
              if ( v68 == -8192 && !v74 )
                v74 = v60;
              v67 = v66 & (v81 + v67);
              v60 = (__int64 *)(v65 + 16LL * v67);
              v68 = *v60;
              if ( v56 == *v60 )
                goto LABEL_60;
              ++v81;
            }
LABEL_66:
            if ( v74 )
              v60 = v74;
          }
        }
        else
        {
          if ( v58 - *(_DWORD *)(a2 + 132) - v61 > v58 >> 3 )
          {
LABEL_41:
            *(_DWORD *)(a2 + 128) = (2 * (v59 >> 1) + 2) | v59 & 1;
            if ( *v60 != -4096 )
              --*(_DWORD *)(a2 + 132);
            *v60 = v56;
            v49 = 0;
            v60[1] = 0;
            goto LABEL_20;
          }
          sub_2ACA3E0(v85, v58);
          v55 = v89;
          if ( (*(_BYTE *)(a2 + 128) & 1) != 0 )
          {
            v69 = a2 + 136;
            v70 = 3;
          }
          else
          {
            v76 = *(_DWORD *)(a2 + 144);
            v69 = *(_QWORD *)(a2 + 136);
            if ( !v76 )
            {
LABEL_111:
              *(_DWORD *)(a2 + 128) = (2 * (*(_DWORD *)(a2 + 128) >> 1) + 2) | *(_DWORD *)(a2 + 128) & 1;
              BUG();
            }
            v70 = v76 - 1;
          }
          v71 = v70 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
          v60 = (__int64 *)(v69 + 16LL * v71);
          v72 = *v60;
          if ( v56 != *v60 )
          {
            v73 = 1;
            v74 = 0;
            while ( v72 != -4096 )
            {
              if ( v72 == -8192 && !v74 )
                v74 = v60;
              v71 = v70 & (v73 + v71);
              v60 = (__int64 *)(v69 + 16LL * v71);
              v72 = *v60;
              if ( v56 == *v60 )
                goto LABEL_60;
              ++v73;
            }
            goto LABEL_66;
          }
        }
LABEL_60:
        v59 = *(_DWORD *)(a2 + 128);
        goto LABEL_41;
      }
LABEL_33:
      v88 = *(_QWORD *)(v88 + 8);
      if ( v88 == v86 )
      {
        while ( 1 )
        {
LABEL_34:
          sub_2AD7320((__int64)v94, v33, v31, v35, v36, v37);
          v32 = v96;
          v31 = v95;
          v33 = v99;
          if ( v96 - v95 == v100 - v99 )
          {
            if ( v95 == v96 )
              goto LABEL_2;
            v77 = v95;
            while ( *(_QWORD *)v77 == *(_QWORD *)v33 )
            {
              v78 = *(_BYTE *)(v77 + 24);
              if ( v78 != *(_BYTE *)(v33 + 24)
                || v78
                && (*(_QWORD *)(v77 + 8) != *(_QWORD *)(v33 + 8) || *(_QWORD *)(v77 + 16) != *(_QWORD *)(v33 + 16)) )
              {
                break;
              }
              v77 += 32;
              v33 += 32LL;
              if ( v96 == v77 )
                goto LABEL_2;
            }
          }
          v35 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v96 - 32) + 8LL) - 1;
          if ( (unsigned int)v35 <= 1 )
            goto LABEL_2;
        }
      }
    }
  }
  if ( v31 != v32 )
  {
    while ( *(_QWORD *)v31 == *(_QWORD *)v33 )
    {
      v79 = *(_BYTE *)(v31 + 24);
      if ( v79 != *(_BYTE *)(v33 + 24)
        || v79 && (*(_QWORD *)(v31 + 8) != *(_QWORD *)(v33 + 8) || *(_QWORD *)(v31 + 16) != *(_QWORD *)(v33 + 16)) )
      {
        break;
      }
      v31 += 32;
      v33 += 32LL;
      if ( v32 == v31 )
        goto LABEL_86;
    }
    goto LABEL_3;
  }
LABEL_86:
  sub_2AB1B50((__int64)v107);
  sub_2AB1B50((__int64)v103);
  sub_2AB1B50((__int64)v98);
  sub_2AB1B50((__int64)v94);
  sub_2AB1B50((__int64)v118);
  sub_2AB1B50((__int64)v116);
  sub_2AB1B50((__int64)v113);
  sub_2AB1B50((__int64)v111);
  sub_2AB1B50((__int64)v93);
  sub_2AB1B50((__int64)v92);
}
