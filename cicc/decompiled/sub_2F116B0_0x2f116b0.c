// Function: sub_2F116B0
// Address: 0x2f116b0
//
void __fastcall sub_2F116B0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 (*v7)(); // rax
  __int64 v8; // r14
  unsigned int v9; // r12d
  __int64 v10; // rax
  unsigned int v11; // edx
  _BYTE *v12; // rax
  char v13; // al
  __int64 v14; // rdi
  _WORD *v15; // rdx
  int v16; // eax
  char *v17; // rsi
  size_t v18; // rdx
  void *v19; // rdi
  __int64 v20; // rax
  char v21; // cl
  unsigned __int8 *v22; // rdx
  unsigned __int64 v23; // r12
  int v24; // esi
  int v25; // ecx
  __int64 v26; // rdi
  _BYTE *v27; // rax
  char v28; // r13
  const char *v29; // r12
  unsigned __int8 *v30; // rdx
  unsigned __int8 v31; // r12
  const char *v32; // r14
  unsigned __int8 *v33; // rdx
  const char *v34; // r14
  unsigned __int8 *v35; // rdx
  unsigned int v36; // r12d
  __int64 v37; // rax
  unsigned int v38; // r12d
  __int64 v39; // rdi
  _BYTE *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rdi
  _BYTE *v44; // rax
  const char *v45; // rax
  _DWORD *v46; // rdx
  int v47; // eax
  __int64 v48; // r12
  __int64 v49; // rax
  int *v50; // rdx
  int v51; // eax
  __int64 v52; // rax
  __int64 *v53; // rbx
  __int64 v54; // r14
  __int64 *v55; // rbx
  __int64 v56; // rax
  _QWORD *v57; // r14
  __int64 v58; // r12
  unsigned __int64 v59; // r12
  __int64 v60; // rdi
  _BYTE *v61; // rax
  __int64 v62; // rax
  unsigned int v63; // edx
  __int64 v64; // rdi
  _WORD *v65; // rdx
  bool v66; // zf
  __int64 v67; // rdi
  _BYTE *v68; // rax
  __int64 v69; // rdi
  _BYTE *v70; // rax
  __int64 v71; // rdi
  _BYTE *v72; // rax
  __int64 v73; // rdi
  _BYTE *v74; // rax
  __int64 v75; // rdi
  _BYTE *v76; // rax
  size_t v77; // [rsp+8h] [rbp-78h]
  __int64 *v78; // [rsp+10h] [rbp-70h]
  __int64 v79; // [rsp+18h] [rbp-68h]
  char v80; // [rsp+20h] [rbp-60h]
  _DWORD *v81; // [rsp+28h] [rbp-58h]
  __int64 v82; // [rsp+28h] [rbp-58h]
  unsigned int v83; // [rsp+30h] [rbp-50h]
  __int64 *v84; // [rsp+30h] [rbp-50h]
  _QWORD *v85; // [rsp+38h] [rbp-48h]
  unsigned __int64 v86[7]; // [rsp+48h] [rbp-38h] BYREF

  v4 = sub_2E88D60(a2);
  v5 = *(_QWORD *)(v4 + 16);
  v6 = *(_QWORD *)(v4 + 32);
  v78 = (__int64 *)v4;
  v85 = 0;
  v81 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 200LL))(v5);
  v7 = *(__int64 (**)())(*(_QWORD *)v5 + 128LL);
  if ( v7 != sub_2DAC790 )
    v85 = (_QWORD *)((__int64 (__fastcall *)(__int64))v7)(v5);
  v8 = 0;
  v9 = 0;
  v86[0] = 0x2000000000000001LL;
  v80 = sub_2E8B990(a2);
  v83 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  if ( v83 )
  {
    while ( 1 )
    {
      v12 = (_BYTE *)(v8 + *(_QWORD *)(a2 + 32));
      if ( *v12 )
        break;
      v13 = v12[3];
      if ( (v13 & 0x10) == 0 || (v13 & 0x20) != 0 )
        break;
      if ( v9 )
      {
        v14 = *a1;
        v15 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v15 <= 1u )
        {
          sub_CB6200(v14, (unsigned __int8 *)", ", 2u);
        }
        else
        {
          *v15 = 8236;
          *(_QWORD *)(v14 + 32) += 2LL;
        }
      }
      v8 += 40;
      v10 = sub_2E8BA90(a2, v9, v86, v6);
      v11 = v9++;
      sub_2F11090(a1, a2, v11, v81, (__int64)v85, v80, v10, 0);
      if ( v83 == v9 )
        goto LABEL_136;
    }
    if ( !v9 )
      goto LABEL_14;
LABEL_136:
    sub_904010(*a1, " = ");
  }
  else
  {
LABEL_14:
    v9 = 0;
  }
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (*(_DWORD *)(a2 + 44) & 1) != 0 )
  {
    sub_904010(*a1, "frame-setup ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
    if ( (*(_BYTE *)(a2 + 44) & 2) == 0 )
    {
LABEL_17:
      if ( (v16 & 0x10) == 0 )
        goto LABEL_18;
      goto LABEL_125;
    }
  }
  else if ( (*(_BYTE *)(a2 + 44) & 2) == 0 )
  {
    goto LABEL_17;
  }
  sub_904010(*a1, "frame-destroy ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a2 + 44) & 0x10) == 0 )
  {
LABEL_18:
    if ( (v16 & 0x20) == 0 )
      goto LABEL_19;
    goto LABEL_126;
  }
LABEL_125:
  sub_904010(*a1, "nnan ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a2 + 44) & 0x20) == 0 )
  {
LABEL_19:
    if ( (v16 & 0x40) == 0 )
      goto LABEL_20;
    goto LABEL_127;
  }
LABEL_126:
  sub_904010(*a1, "ninf ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (*(_BYTE *)(a2 + 44) & 0x40) == 0 )
  {
LABEL_20:
    if ( (v16 & 0x80u) == 0 )
      goto LABEL_21;
    goto LABEL_128;
  }
LABEL_127:
  sub_904010(*a1, "nsz ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( *(char *)(a2 + 44) >= 0 )
  {
LABEL_21:
    if ( (v16 & 0x100) == 0 )
      goto LABEL_22;
    goto LABEL_129;
  }
LABEL_128:
  sub_904010(*a1, "arcp ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x100) == 0 )
  {
LABEL_22:
    if ( (v16 & 0x200) == 0 )
      goto LABEL_23;
    goto LABEL_130;
  }
LABEL_129:
  sub_904010(*a1, "contract ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x200) == 0 )
  {
LABEL_23:
    if ( (v16 & 0x400) == 0 )
      goto LABEL_24;
    goto LABEL_131;
  }
LABEL_130:
  sub_904010(*a1, "afn ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x400) == 0 )
  {
LABEL_24:
    if ( (v16 & 0x800) == 0 )
      goto LABEL_25;
    goto LABEL_132;
  }
LABEL_131:
  sub_904010(*a1, "reassoc ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x800) == 0 )
  {
LABEL_25:
    if ( (v16 & 0x1000) == 0 )
      goto LABEL_26;
    goto LABEL_133;
  }
LABEL_132:
  sub_904010(*a1, "nuw ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x1000) == 0 )
  {
LABEL_26:
    if ( (v16 & 0x2000) == 0 )
      goto LABEL_27;
LABEL_134:
    sub_904010(*a1, "exact ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
    if ( (v16 & 0x4000) == 0 )
      goto LABEL_28;
    goto LABEL_135;
  }
LABEL_133:
  sub_904010(*a1, "nsw ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  if ( (v16 & 0x2000) != 0 )
    goto LABEL_134;
LABEL_27:
  if ( (v16 & 0x4000) == 0 )
    goto LABEL_28;
LABEL_135:
  sub_904010(*a1, "nofpexcept ");
  v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
LABEL_28:
  if ( (v16 & 0x8000) != 0 )
  {
    sub_904010(*a1, "nomerge ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x10000) != 0 )
  {
    sub_904010(*a1, "unpredictable ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x20000) != 0 )
  {
    sub_904010(*a1, "noconvergent ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x40000) != 0 )
  {
    sub_904010(*a1, "nneg ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x80000) != 0 )
  {
    sub_904010(*a1, "disjoint ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x100000) != 0 )
  {
    sub_904010(*a1, "nusw ");
    v16 = *(_DWORD *)(a2 + 44) & 0xFFFFFF;
  }
  if ( (v16 & 0x200000) != 0 )
    sub_904010(*a1, "samesign ");
  v17 = (char *)(v85[3] + *(unsigned int *)(v85[2] + 4LL * *(unsigned __int16 *)(a2 + 68)));
  if ( !v17 )
  {
LABEL_46:
    if ( v83 <= v9 )
      goto LABEL_47;
LABEL_109:
    v60 = *a1;
    v61 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v61 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v60, 32);
    }
    else
    {
      *(_QWORD *)(v60 + 32) = v61 + 1;
      *v61 = 32;
    }
    while ( 1 )
    {
      v62 = sub_2E8BA90(a2, v9, v86, v6);
      v63 = v9++;
      sub_2F11090(a1, a2, v63, v81, (__int64)v85, v80, v62, 1);
      if ( v9 == v83 )
        break;
      v64 = *a1;
      v65 = *(_WORD **)(*a1 + 32);
      if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v65 <= 1u )
      {
        sub_CB6200(v64, (unsigned __int8 *)", ", 2u);
      }
      else
      {
        *v65 = 8236;
        *(_QWORD *)(v64 + 32) += 2LL;
      }
    }
    v20 = *(_QWORD *)(a2 + 48);
    v22 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v38 = *(_DWORD *)(a2 + 64);
      v28 = 1;
      if ( v38 )
      {
LABEL_76:
        v39 = *a1;
        v40 = *(_BYTE **)(*a1 + 32);
        if ( (unsigned __int64)v40 >= *(_QWORD *)(*a1 + 24) )
        {
          sub_CB5D20(v39, 44);
        }
        else
        {
          *(_QWORD *)(v39 + 32) = v40 + 1;
          *v40 = 44;
        }
        goto LABEL_78;
      }
LABEL_152:
      if ( !(_BYTE)qword_50224E8 || !*(_QWORD *)(a2 + 56) )
        goto LABEL_85;
      v42 = a2 + 56;
      if ( !v28 )
        goto LABEL_83;
      goto LABEL_81;
    }
    v21 = 1;
    goto LABEL_48;
  }
  v79 = *a1;
  v18 = strlen(v17);
  v19 = *(void **)(v79 + 32);
  if ( *(_QWORD *)(v79 + 24) - (_QWORD)v19 >= v18 )
  {
    if ( v18 )
    {
      v77 = v18;
      memcpy(v19, v17, v18);
      *(_QWORD *)(v79 + 32) += v77;
    }
    goto LABEL_46;
  }
  sub_CB6200(v79, (unsigned __int8 *)v17, v18);
  if ( v83 > v9 )
    goto LABEL_109;
LABEL_47:
  v20 = *(_QWORD *)(a2 + 48);
  v21 = 0;
  v22 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    v38 = *(_DWORD *)(a2 + 64);
    if ( !v38 )
    {
      if ( !(_BYTE)qword_50224E8 )
        goto LABEL_85;
      v42 = a2 + 56;
      if ( !*(_QWORD *)(a2 + 56) )
        goto LABEL_85;
      goto LABEL_83;
    }
LABEL_78:
    v41 = sub_904010(*a1, " debug-instr-number ");
    sub_CB59D0(v41, v38);
    if ( !(_BYTE)qword_50224E8 )
      goto LABEL_84;
    goto LABEL_79;
  }
LABEL_48:
  v23 = (unsigned __int64)v22;
  v24 = v20 & 7;
  if ( v24 != 1 )
  {
    if ( v24 != 3 )
    {
      if ( v24 != 2 )
        goto LABEL_162;
LABEL_106:
      if ( !v21 )
      {
LABEL_54:
        sub_904010(*a1, " post-instr-symbol ");
        sub_2EABE30(*a1, v23);
        v20 = *(_QWORD *)(a2 + 48);
        v22 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          goto LABEL_185;
        v21 = 1;
        goto LABEL_56;
      }
LABEL_52:
      v26 = *a1;
      v27 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v27 >= *(_QWORD *)(*a1 + 24) )
      {
        sub_CB5D20(v26, 44);
      }
      else
      {
        *(_QWORD *)(v26 + 32) = v27 + 1;
        *v27 = 44;
      }
      goto LABEL_54;
    }
    if ( !v22[4] || (v23 = *(_QWORD *)&v22[8 * *(int *)v22 + 16]) == 0 )
    {
      if ( !v22[5] )
        goto LABEL_162;
LABEL_105:
      v23 = *(_QWORD *)&v22[8 * v22[4] + 16 + 8 * (__int64)*(int *)v22];
      if ( !v23 )
        goto LABEL_162;
      goto LABEL_106;
    }
  }
  if ( v21 )
  {
    v67 = *a1;
    v68 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v68 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v67, 44);
    }
    else
    {
      *(_QWORD *)(v67 + 32) = v68 + 1;
      *v68 = 44;
    }
  }
  sub_904010(*a1, " pre-instr-symbol ");
  sub_2EABE30(*a1, v23);
  v20 = *(_QWORD *)(a2 + 48);
  v22 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
LABEL_185:
    v38 = *(_DWORD *)(a2 + 64);
    if ( !v38 )
    {
      if ( !(_BYTE)qword_50224E8 || !*(_QWORD *)(a2 + 56) )
        goto LABEL_85;
      goto LABEL_80;
    }
    goto LABEL_76;
  }
  v23 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v25 = v20 & 7;
  if ( v25 == 2 )
    goto LABEL_52;
  v66 = v25 == 3;
  v21 = 1;
  if ( !v66 )
    goto LABEL_56;
  if ( v22[5] )
    goto LABEL_105;
LABEL_162:
  v22 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_163;
LABEL_56:
  if ( (v20 & 7) != 3
    || (v28 = v22[6]) == 0
    || (v29 = *(const char **)&v22[8 * *(int *)v22 + 16 + 8 * (__int64)(v22[5] + v22[4])]) == 0 )
  {
    v30 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
    {
      v28 = v21;
      goto LABEL_61;
    }
LABEL_163:
    v28 = v21;
    goto LABEL_120;
  }
  if ( v21 )
  {
    v69 = *a1;
    v70 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v70 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v69, 44);
    }
    else
    {
      *(_QWORD *)(v69 + 32) = v70 + 1;
      *v70 = 44;
    }
  }
  sub_904010(*a1, " heap-alloc-marker ");
  sub_A61DC0(v29, *a1, a1[1], 0);
  v20 = *(_QWORD *)(a2 + 48);
  v30 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_185;
LABEL_61:
  if ( (v20 & 7) != 3 )
  {
    v33 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_141;
  }
  v31 = v30[7];
  if ( !v31 || (v32 = *(const char **)&v30[8 * v30[6] + 16 + 8 * *(int *)v30 + 8 * (__int64)(v30[5] + v30[4])]) == 0 )
  {
    v33 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_120;
LABEL_141:
    v31 = v28;
    if ( (v20 & 7) == 3 )
      goto LABEL_67;
LABEL_142:
    v28 = v31;
    v35 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_71;
  }
  if ( v28 )
  {
    v73 = *a1;
    v74 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v74 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v73, 44);
    }
    else
    {
      *(_QWORD *)(v73 + 32) = v74 + 1;
      *v74 = 44;
    }
  }
  sub_904010(*a1, " pcsections ");
  sub_A61DC0(v32, *a1, a1[1], 0);
  v20 = *(_QWORD *)(a2 + 48);
  v33 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_185;
  if ( (v20 & 7) != 3 )
    goto LABEL_142;
LABEL_67:
  v28 = v33[9];
  if ( v28
    && (v34 = *(const char **)&v33[8 * v33[7] + 16 + 8 * v33[6] + 8 * *(int *)v33 + 8 * (__int64)(v33[5] + v33[4])]) != 0 )
  {
    if ( v31 )
    {
      v71 = *a1;
      v72 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v72 >= *(_QWORD *)(*a1 + 24) )
      {
        sub_CB5D20(v71, 44);
      }
      else
      {
        *(_QWORD *)(v71 + 32) = v72 + 1;
        *v72 = 44;
      }
    }
    sub_904010(*a1, " mmra ");
    sub_A61DC0(v34, *a1, a1[1], 0);
    v20 = *(_QWORD *)(a2 + 48);
    v35 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    {
      v38 = *(_DWORD *)(a2 + 64);
      if ( !v38 )
      {
        if ( !(_BYTE)qword_50224E8 || !*(_QWORD *)(a2 + 56) )
          goto LABEL_95;
        goto LABEL_80;
      }
      goto LABEL_76;
    }
  }
  else
  {
    v28 = v31;
    v35 = (unsigned __int8 *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
    if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
      goto LABEL_120;
  }
LABEL_71:
  if ( (v20 & 7) != 3
    || !v35[8]
    || (v36 = *(_DWORD *)&v35[8 * *(int *)v35 + 16 + 8 * v35[7] + 8 * v35[6] + 8 * (__int64)(v35[5] + v35[4])]) == 0 )
  {
LABEL_120:
    v38 = *(_DWORD *)(a2 + 64);
    if ( v38 )
    {
      if ( !v28 )
        goto LABEL_78;
      goto LABEL_76;
    }
    goto LABEL_152;
  }
  if ( v28 )
  {
    v75 = *a1;
    v76 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v76 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v75, 44);
    }
    else
    {
      *(_QWORD *)(v75 + 32) = v76 + 1;
      *v76 = 44;
    }
  }
  v37 = sub_904010(*a1, " cfi-type ");
  sub_CB59D0(v37, v36);
  v38 = *(_DWORD *)(a2 + 64);
  if ( v38 )
    goto LABEL_76;
  if ( !(_BYTE)qword_50224E8 )
    goto LABEL_84;
LABEL_79:
  if ( *(_QWORD *)(a2 + 56) )
  {
LABEL_80:
    v42 = a2 + 56;
LABEL_81:
    v43 = *a1;
    v44 = *(_BYTE **)(*a1 + 32);
    if ( (unsigned __int64)v44 >= *(_QWORD *)(*a1 + 24) )
    {
      sub_CB5D20(v43, 44);
    }
    else
    {
      *(_QWORD *)(v43 + 32) = v44 + 1;
      *v44 = 44;
    }
LABEL_83:
    sub_904010(*a1, " debug-location ");
    v45 = (const char *)sub_B10CD0(v42);
    sub_A61DC0(v45, *a1, a1[1], 0);
  }
LABEL_84:
  v20 = *(_QWORD *)(a2 + 48);
LABEL_85:
  v46 = (_DWORD *)(v20 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
    goto LABEL_95;
  v47 = v20 & 7;
  if ( v47 )
  {
    if ( v47 != 3 || !*v46 )
      goto LABEL_95;
  }
  else
  {
    *(_QWORD *)(a2 + 48) = v46;
  }
  sub_904010(*a1, " :: ");
  v48 = sub_B2BE50(*v78);
  v49 = *(_QWORD *)(a2 + 48);
  v50 = (int *)(v49 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v49 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v82 = v78[6];
    v51 = v49 & 7;
    if ( v51 )
    {
      if ( v51 != 3 )
        goto LABEL_95;
      v53 = (__int64 *)(v50 + 4);
      v52 = *v50;
    }
    else
    {
      *(_QWORD *)(a2 + 48) = v50;
      v52 = 1;
      v53 = (__int64 *)(a2 + 48);
    }
    v84 = &v53[v52];
    if ( &v53[v52] != v53 )
    {
      v54 = *v53;
      v55 = v53 + 1;
      v56 = v54;
      v57 = (_QWORD *)v48;
      v58 = v56;
      while ( 1 )
      {
        sub_2EAC530(v58, *a1, a1[1], (__int64)(a1 + 4), v57, v82, v85);
        if ( v84 == v55 )
          break;
        v58 = *v55++;
        sub_904010(*a1, ", ");
      }
    }
  }
LABEL_95:
  v59 = v86[0];
  if ( (v86[0] & 1) == 0 && v86[0] )
  {
    if ( *(_QWORD *)v86[0] != v86[0] + 16 )
      _libc_free(*(_QWORD *)v86[0]);
    j_j___libc_free_0(v59);
  }
}
