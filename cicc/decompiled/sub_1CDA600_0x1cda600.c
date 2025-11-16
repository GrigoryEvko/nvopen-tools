// Function: sub_1CDA600
// Address: 0x1cda600
//
__int64 __fastcall sub_1CDA600(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r15
  __int64 v4; // rcx
  unsigned int v5; // edx
  unsigned __int64 v6; // rax
  __int64 v7; // r8
  unsigned int v8; // r12d
  __int64 v9; // rbx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rsi
  int v13; // ecx
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  int v18; // esi
  unsigned int v19; // r9d
  int v20; // r11d
  unsigned int v21; // edx
  unsigned int v22; // r13d
  unsigned __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // r10
  float v26; // xmm0_4
  unsigned int v27; // esi
  int v28; // ecx
  int v29; // eax
  unsigned int v30; // esi
  float v31; // xmm2_4
  float v32; // xmm1_4
  __int64 v33; // rdi
  unsigned int v34; // edx
  unsigned __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // r12
  __int64 v38; // rbx
  int v39; // r14d
  __int64 v40; // r13
  __int64 v41; // r12
  int v42; // r8d
  __int64 v43; // r9
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  int v47; // r13d
  _BYTE *v48; // rdi
  int v49; // eax
  __int64 v50; // r13
  __int64 v51; // rsi
  __int64 v52; // rsi
  unsigned int v53; // esi
  __int64 v54; // rcx
  unsigned int v55; // edx
  unsigned __int64 v56; // rax
  int v57; // edi
  __int64 v58; // rdi
  __int64 v60; // rax
  int v61; // r8d
  __int64 v62; // rax
  unsigned int v63; // ecx
  __int64 v64; // rsi
  unsigned int v65; // edx
  __int64 v66; // rdi
  __int64 v67; // r13
  int v68; // r10d
  unsigned __int64 v69; // r9
  int v70; // ecx
  __int64 v71; // rax
  _QWORD *v72; // rax
  _QWORD *i; // rdx
  int v74; // r8d
  int v75; // r11d
  unsigned __int64 v76; // r9
  int v77; // ecx
  int v78; // ecx
  __int64 v79; // rdi
  int v80; // r13d
  unsigned __int64 v81; // r10
  int v82; // r13d
  unsigned __int64 v83; // r10
  int v84; // edx
  float *v85; // r11
  __int64 v86; // [rsp+8h] [rbp-D8h]
  __int64 v87; // [rsp+10h] [rbp-D0h]
  __int64 v88; // [rsp+18h] [rbp-C8h]
  __int64 v89; // [rsp+20h] [rbp-C0h]
  __int64 v90; // [rsp+30h] [rbp-B0h]
  float v91; // [rsp+38h] [rbp-A8h]
  float v92; // [rsp+38h] [rbp-A8h]
  float v93; // [rsp+38h] [rbp-A8h]
  __int64 v94; // [rsp+38h] [rbp-A8h]
  __int64 v95; // [rsp+48h] [rbp-98h] BYREF
  _BYTE *v96; // [rsp+50h] [rbp-90h] BYREF
  __int64 v97; // [rsp+58h] [rbp-88h]
  _BYTE v98[16]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v99; // [rsp+70h] [rbp-70h] BYREF
  __int64 v100; // [rsp+78h] [rbp-68h]
  __int64 v101; // [rsp+80h] [rbp-60h]
  unsigned int v102; // [rsp+88h] [rbp-58h]
  _BYTE *v103; // [rsp+90h] [rbp-50h] BYREF
  __int64 v104; // [rsp+98h] [rbp-48h]
  _BYTE v105[64]; // [rsp+A0h] [rbp-40h] BYREF

  v2 = a1;
  v3 = *(_QWORD *)(a2 + 80);
  v90 = a2 + 72;
  if ( v3 )
  {
    v99 = 0;
    v89 = v3 - 24;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    if ( v3 == a2 + 72 )
    {
      v58 = 0;
      return j___libc_free_0(v58);
    }
  }
  else
  {
    v99 = 0;
    v100 = 0;
    v101 = 0;
    v102 = 0;
    v89 = 0;
  }
  v86 = a1 + 280;
  while ( 2 )
  {
    v8 = dword_4FC0320;
    v9 = v3 - 24;
    if ( !v3 )
      v9 = 0;
    if ( !dword_4FC0320 )
      goto LABEL_9;
    v103 = v105;
    v104 = 0x200000000LL;
    v97 = 0x200000000LL;
    v96 = v98;
    v37 = *(_QWORD *)(v9 + 48);
    if ( v9 + 40 == v37 )
    {
      v8 = 0;
      goto LABEL_55;
    }
    v88 = v9;
    v38 = v9 + 40;
    v87 = v2;
    v39 = 0;
    v40 = v37;
    do
    {
      v41 = v40 - 24;
      if ( !v40 )
        v41 = 0;
      ++v39;
      if ( (unsigned __int8)sub_1C30710(v41) )
        goto LABEL_46;
      if ( *(_BYTE *)(v41 + 16) == 54 )
      {
        v44 = **(_QWORD **)(v41 - 24);
        if ( *(_BYTE *)(v44 + 8) == 16 )
          v44 = **(_QWORD **)(v44 + 16);
        if ( ((*(_DWORD *)(v44 + 8) >> 8) & 0xFFFFFFFB) == 1 )
        {
LABEL_46:
          v45 = (unsigned int)v104;
          if ( (_DWORD)v104 )
          {
            v46 = (unsigned int)v97;
            if ( (unsigned int)v97 >= HIDWORD(v97) )
            {
              sub_16CD150((__int64)&v96, v98, 0, 4, v42, v43);
              v46 = (unsigned int)v97;
            }
            *(_DWORD *)&v96[4 * v46] = v39;
            v45 = (unsigned int)v104;
            LODWORD(v97) = v97 + 1;
            if ( HIDWORD(v104) > (unsigned int)v104 )
              goto LABEL_37;
          }
          else if ( HIDWORD(v104) > (unsigned int)v104 )
          {
LABEL_37:
            v39 = 0;
            *(_QWORD *)&v103[8 * v45] = v41;
            LODWORD(v104) = v104 + 1;
            goto LABEL_38;
          }
          sub_16CD150((__int64)&v103, v105, 0, 8, v42, v43);
          v45 = (unsigned int)v104;
          goto LABEL_37;
        }
      }
LABEL_38:
      v40 = *(_QWORD *)(v40 + 8);
    }
    while ( v38 != v40 );
    v8 = v104;
    v47 = v39;
    v9 = v88;
    v2 = v87;
    if ( !(_DWORD)v104 )
      goto LABEL_52;
    v60 = (unsigned int)v97;
    if ( (unsigned int)v97 >= HIDWORD(v97) )
    {
      sub_16CD150((__int64)&v96, v98, 0, 4, v42, v43);
      v60 = (unsigned int)v97;
    }
    *(_DWORD *)&v96[4 * v60] = v47;
    v61 = v104;
    LODWORD(v97) = v97 + 1;
    if ( !(_DWORD)v104 )
      goto LABEL_94;
    v62 = 0;
    v63 = 0;
    v8 = 0;
    v64 = 4LL * (unsigned int)v104;
    do
    {
      v65 = *(_DWORD *)&v96[v62];
      v63 += v65;
      if ( v65 <= 9 )
      {
        v66 = v8++;
        v43 = *(_QWORD *)&v103[2 * v62];
        *(_QWORD *)&v103[8 * v66] = v43;
      }
      v62 += 4;
    }
    while ( v64 != v62 );
    if ( 10 * v61 > v63 )
    {
      if ( v8 == v61 )
        goto LABEL_86;
      v71 = (unsigned int)v104;
      if ( v8 < (unsigned __int64)(unsigned int)v104 )
      {
LABEL_104:
        LODWORD(v104) = v8;
      }
      else if ( v8 > (unsigned __int64)(unsigned int)v104 )
      {
        if ( v8 > (unsigned __int64)HIDWORD(v104) )
        {
          sub_16CD150((__int64)&v103, v105, v8, 8, v61, v43);
          v71 = (unsigned int)v104;
        }
        v72 = &v103[8 * v71];
        for ( i = &v103[8 * v8]; i != v72; ++v72 )
        {
          if ( v72 )
            *v72 = 0;
        }
        goto LABEL_104;
      }
      if ( v8 )
      {
LABEL_86:
        v67 = *(_QWORD *)v103;
        v95 = v88;
        sub_1CD3CC0(v87 + 344, &v95)[1] = v67;
      }
LABEL_52:
      v48 = v96;
    }
    else
    {
LABEL_94:
      v48 = v96;
      v8 = 0;
    }
    if ( v48 != v98 )
      _libc_free((unsigned __int64)v48);
LABEL_55:
    if ( v103 != v105 )
      _libc_free((unsigned __int64)v103);
LABEL_9:
    v10 = *(_QWORD *)(v2 + 192);
    v11 = *(_DWORD *)(v10 + 24);
    if ( !v11 )
      goto LABEL_28;
    v12 = *(_QWORD *)(v10 + 8);
    v13 = v11 - 1;
    v14 = (v11 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v15 = (__int64 *)(v12 + 16LL * v14);
    v16 = *v15;
    if ( v9 != *v15 )
    {
      v29 = 1;
      while ( v16 != -8 )
      {
        v74 = v29 + 1;
        v14 = v13 & (v29 + v14);
        v15 = (__int64 *)(v12 + 16LL * v14);
        v16 = *v15;
        if ( v9 == *v15 )
          goto LABEL_11;
        v29 = v74;
      }
LABEL_28:
      v95 = 0;
      goto LABEL_29;
    }
LABEL_11:
    v17 = v15[1];
    v95 = v17;
    if ( v17 )
    {
      v18 = v102;
      if ( !v102 )
        goto LABEL_59;
      v19 = v102 - 1;
      v20 = 1;
      v21 = (v102 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v22 = v21;
      v23 = v100 + 16LL * v21;
      v24 = *(_QWORD *)v23;
      v25 = *(_QWORD *)v23;
      if ( v17 == *(_QWORD *)v23 )
      {
        if ( v23 != v100 + 16LL * v102 )
        {
          v26 = *(float *)(v23 + 8);
LABEL_16:
          if ( !sub_15CCCD0(*(_QWORD *)(v2 + 176), v9, **(_QWORD **)(v17 + 32)) )
            v26 = v26 * 0.5;
LABEL_18:
          if ( !v8 )
            goto LABEL_19;
LABEL_31:
          v30 = *(_DWORD *)(v2 + 336);
          v96 = (_BYTE *)v9;
          v31 = (float)(int)(dword_4FC0320 * v8);
          v32 = v26 / v31;
          if ( v30 )
          {
            v33 = *(_QWORD *)(v2 + 320);
            v34 = (v30 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v35 = v33 + 16LL * v34;
            v36 = *(_QWORD *)v35;
            if ( v9 == *(_QWORD *)v35 )
            {
LABEL_33:
              *(float *)(v35 + 8) = v32;
              goto LABEL_19;
            }
            v75 = 1;
            v76 = 0;
            while ( v36 != -8 )
            {
              if ( v36 == -16 && !v76 )
                v76 = v35;
              v34 = (v30 - 1) & (v75 + v34);
              v35 = v33 + 16LL * v34;
              v36 = *(_QWORD *)v35;
              if ( v9 == *(_QWORD *)v35 )
                goto LABEL_33;
              ++v75;
            }
            v77 = *(_DWORD *)(v2 + 328);
            if ( v76 )
              v35 = v76;
            ++*(_QWORD *)(v2 + 312);
            v78 = v77 + 1;
            if ( 4 * v78 < 3 * v30 )
            {
              v79 = v9;
              if ( v30 - *(_DWORD *)(v2 + 332) - v78 <= v30 >> 3 )
              {
                sub_1CD3880(v2 + 312, v30);
                sub_1CD2F90(v2 + 312, (__int64 *)&v96, &v103);
                v35 = (unsigned __int64)v103;
                v79 = (__int64)v96;
                v32 = v26 / v31;
                v78 = *(_DWORD *)(v2 + 328) + 1;
              }
              goto LABEL_121;
            }
          }
          else
          {
            ++*(_QWORD *)(v2 + 312);
          }
          sub_1CD3880(v2 + 312, 2 * v30);
          sub_1CD2F90(v2 + 312, (__int64 *)&v96, &v103);
          v35 = (unsigned __int64)v103;
          v79 = (__int64)v96;
          v32 = v26 / v31;
          v78 = *(_DWORD *)(v2 + 328) + 1;
LABEL_121:
          *(_DWORD *)(v2 + 328) = v78;
          if ( *(_QWORD *)v35 != -8 )
            --*(_DWORD *)(v2 + 332);
          *(_QWORD *)v35 = v79;
          *(_DWORD *)(v35 + 8) = 0;
          goto LABEL_33;
        }
LABEL_59:
        v49 = dword_4FBFFA0;
        v50 = *(_QWORD *)v17;
        v26 = (float)dword_4FBFFA0;
        if ( *(_QWORD *)v17 )
        {
          while ( 1 )
          {
            v92 = v26 * (float)v49;
            v26 = v92;
            v51 = sub_13FC520(v17);
            if ( !v51 || (v26 = v92, !sub_15CCCD0(*(_QWORD *)(v2 + 176), v51, **(_QWORD **)(v50 + 32))) )
              v26 = v26 * 0.5;
            if ( !*(_QWORD *)v50 )
              break;
            v17 = v50;
            v50 = *(_QWORD *)v50;
            v49 = dword_4FBFFA0;
          }
        }
        else
        {
          v50 = v17;
        }
        v52 = sub_13FC520(v50);
        if ( v52 && sub_15CCCD0(*(_QWORD *)(v2 + 176), v52, v89) )
        {
          v53 = v102;
          if ( v102 )
            goto LABEL_69;
LABEL_72:
          ++v99;
LABEL_73:
          v93 = v26;
          v53 *= 2;
LABEL_74:
          sub_1CDA470((__int64)&v99, v53);
          sub_1CD3670((__int64)&v99, &v95, &v103);
          v56 = (unsigned __int64)v103;
          v54 = v95;
          v26 = v93;
          v57 = v101 + 1;
          goto LABEL_134;
        }
        v53 = v102;
        v26 = v26 * 0.5;
        if ( !v102 )
          goto LABEL_72;
LABEL_69:
        v54 = v95;
        v55 = (v53 - 1) & (((unsigned int)v95 >> 9) ^ ((unsigned int)v95 >> 4));
        v56 = v100 + 16LL * v55;
        v17 = *(_QWORD *)v56;
        if ( v95 != *(_QWORD *)v56 )
        {
          v80 = 1;
          v81 = 0;
          while ( v17 != -8 )
          {
            if ( !v81 && v17 == -16 )
              v81 = v56;
            v55 = (v53 - 1) & (v80 + v55);
            v56 = v100 + 16LL * v55;
            v17 = *(_QWORD *)v56;
            if ( v95 == *(_QWORD *)v56 )
              goto LABEL_70;
            ++v80;
          }
          if ( v81 )
            v56 = v81;
          ++v99;
          v57 = v101 + 1;
          if ( 4 * ((int)v101 + 1) >= 3 * v53 )
            goto LABEL_73;
          if ( v53 - HIDWORD(v101) - v57 <= v53 >> 3 )
          {
            v93 = v26;
            goto LABEL_74;
          }
LABEL_134:
          LODWORD(v101) = v57;
          if ( *(_QWORD *)v56 != -8 )
            --HIDWORD(v101);
          *(_QWORD *)v56 = v54;
          *(_DWORD *)(v56 + 8) = 0;
          v17 = v95;
        }
LABEL_70:
        *(float *)(v56 + 8) = v26;
        goto LABEL_16;
      }
      while ( 1 )
      {
        if ( v25 == -8 )
          goto LABEL_59;
        v22 = v19 & (v20 + v22);
        v25 = *(_QWORD *)(v100 + 16LL * v22);
        v94 = v100 + 16LL * v22;
        if ( v17 == v25 )
          break;
        ++v20;
      }
      v82 = 1;
      v83 = 0;
      if ( v94 == v100 + 16LL * v102 )
        goto LABEL_59;
      while ( v24 != -8 )
      {
        if ( v24 != -16 || v83 )
          v23 = v83;
        v21 = v19 & (v82 + v21);
        v85 = (float *)(v100 + 16LL * v21);
        v24 = *(_QWORD *)v85;
        if ( v17 == *(_QWORD *)v85 )
        {
          v26 = v85[2];
          goto LABEL_16;
        }
        ++v82;
        v83 = v23;
        v23 = v100 + 16LL * v21;
      }
      if ( !v83 )
        v83 = v23;
      ++v99;
      v84 = v101 + 1;
      if ( 4 * ((int)v101 + 1) >= 3 * v102 )
      {
        v18 = 2 * v102;
      }
      else if ( v102 - HIDWORD(v101) - v84 > v102 >> 3 )
      {
LABEL_145:
        LODWORD(v101) = v84;
        if ( *(_QWORD *)v83 != -8 )
          --HIDWORD(v101);
        *(_QWORD *)v83 = v17;
        v26 = 0.0;
        *(_DWORD *)(v83 + 8) = 0;
        v17 = v95;
        goto LABEL_16;
      }
      sub_1CDA470((__int64)&v99, v18);
      sub_1CD3670((__int64)&v99, &v95, &v103);
      v83 = (unsigned __int64)v103;
      v17 = v95;
      v84 = v101 + 1;
      goto LABEL_145;
    }
LABEL_29:
    v26 = 1.0;
    if ( sub_15CCCD0(*(_QWORD *)(v2 + 176), v9, v89) )
      goto LABEL_18;
    v26 = 0.5;
    if ( v8 )
      goto LABEL_31;
LABEL_19:
    v27 = *(_DWORD *)(v2 + 304);
    v96 = (_BYTE *)v9;
    if ( !v27 )
    {
      ++*(_QWORD *)(v2 + 280);
      goto LABEL_21;
    }
    v4 = *(_QWORD *)(v2 + 288);
    v5 = (v27 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v6 = v4 + 16LL * v5;
    v7 = *(_QWORD *)v6;
    if ( v9 != *(_QWORD *)v6 )
    {
      v68 = 1;
      v69 = 0;
      while ( v7 != -8 )
      {
        if ( !v69 && v7 == -16 )
          v69 = v6;
        v5 = (v27 - 1) & (v68 + v5);
        v6 = v4 + 16LL * v5;
        v7 = *(_QWORD *)v6;
        if ( v9 == *(_QWORD *)v6 )
          goto LABEL_5;
        ++v68;
      }
      v70 = *(_DWORD *)(v2 + 296);
      if ( v69 )
        v6 = v69;
      ++*(_QWORD *)(v2 + 280);
      v28 = v70 + 1;
      if ( 4 * v28 < 3 * v27 )
      {
        if ( v27 - *(_DWORD *)(v2 + 300) - v28 <= v27 >> 3 )
        {
          v91 = v26;
LABEL_22:
          sub_1CD3880(v86, v27);
          sub_1CD2F90(v86, (__int64 *)&v96, &v103);
          v6 = (unsigned __int64)v103;
          v9 = (__int64)v96;
          v26 = v91;
          v28 = *(_DWORD *)(v2 + 296) + 1;
        }
        *(_DWORD *)(v2 + 296) = v28;
        if ( *(_QWORD *)v6 != -8 )
          --*(_DWORD *)(v2 + 300);
        *(_QWORD *)v6 = v9;
        *(_DWORD *)(v6 + 8) = 0;
        goto LABEL_5;
      }
LABEL_21:
      v91 = v26;
      v27 *= 2;
      goto LABEL_22;
    }
LABEL_5:
    *(float *)(v6 + 8) = v26;
    v3 = *(_QWORD *)(v3 + 8);
    if ( v3 != v90 )
      continue;
    break;
  }
  v58 = v100;
  return j___libc_free_0(v58);
}
