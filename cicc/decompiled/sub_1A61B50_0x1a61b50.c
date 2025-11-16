// Function: sub_1A61B50
// Address: 0x1a61b50
//
__int64 __fastcall sub_1A61B50(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  char v10; // r12
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int8 v26; // al
  __int64 v27; // rsi
  unsigned int v28; // eax
  __int64 *v29; // r12
  unsigned __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  double v33; // xmm4_8
  double v34; // xmm5_8
  __int64 v35; // rax
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 v38; // r15
  int v39; // eax
  __int64 v40; // rax
  int v41; // edx
  __int64 v42; // rdx
  __int64 **v43; // rax
  __int64 *v44; // rcx
  unsigned __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  _QWORD *v50; // rax
  _QWORD *v51; // rdi
  __int64 v52; // r13
  __int64 v53; // r14
  __int64 v54; // rdi
  unsigned __int64 v55; // rbx
  _QWORD *v56; // rcx
  _QWORD *i; // rax
  char v58; // dl
  unsigned int v59; // edx
  _QWORD *v60; // rcx
  __int64 v62; // rdx
  __int64 v63; // r12
  int v64; // r13d
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // r13
  _QWORD *v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r9
  __int64 *v72; // r15
  __int64 v73; // r14
  unsigned __int64 v74; // rbx
  __int64 v75; // rcx
  __int64 **v76; // rax
  __int64 *v77; // rdi
  unsigned __int64 v78; // rcx
  __int64 v79; // rcx
  __int64 v80; // rcx
  __int64 v81; // rdi
  __int64 v82; // r8
  int v83; // eax
  __int64 v84; // rax
  int v85; // ecx
  unsigned __int64 v86; // rax
  __int64 *v87; // rax
  unsigned __int64 v88; // rdx
  __int64 v89; // rdx
  __int64 v90; // rax
  __int64 v91; // [rsp+0h] [rbp-C0h]
  __int64 v92; // [rsp+8h] [rbp-B8h]
  __int64 v93; // [rsp+10h] [rbp-B0h]
  __int64 v94; // [rsp+18h] [rbp-A8h]
  __int64 v95; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v96; // [rsp+28h] [rbp-98h]
  int v97; // [rsp+28h] [rbp-98h]
  unsigned __int64 v98; // [rsp+28h] [rbp-98h]
  __int64 *v99; // [rsp+30h] [rbp-90h]
  __int64 v100; // [rsp+30h] [rbp-90h]
  __int64 **v101; // [rsp+38h] [rbp-88h]
  __int64 v102; // [rsp+38h] [rbp-88h]
  __int64 v103; // [rsp+38h] [rbp-88h]
  unsigned __int8 v105; // [rsp+4Fh] [rbp-71h]
  __int64 v107; // [rsp+58h] [rbp-68h]
  __int64 v108; // [rsp+60h] [rbp-60h]
  __int64 v109; // [rsp+68h] [rbp-58h]
  char *v110; // [rsp+70h] [rbp-50h] BYREF
  char v111; // [rsp+80h] [rbp-40h]
  char v112; // [rsp+81h] [rbp-3Fh]

  if ( (unsigned __int8)sub_1636880(a1, a2) )
    return 0;
  v10 = 0;
  if ( *(_QWORD *)(a1 + 200) )
  {
    if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64))(a1 + 208))(a1 + 184, a2) )
      return 0;
  }
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_128:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F9D764 )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_128;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F9D764);
  v15 = sub_14CF090(v14, a2);
  v16 = *(__int64 **)(a1 + 8);
  *(_QWORD *)(a1 + 176) = v15;
  v17 = *v16;
  v18 = v16[1];
  if ( v17 == v18 )
LABEL_129:
    BUG();
  while ( *(_UNKNOWN **)v17 != &unk_4F9D3C0 )
  {
    v17 += 16;
    if ( v18 == v17 )
      goto LABEL_129;
  }
  v19 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v17 + 8) + 104LL))(*(_QWORD *)(v17 + 8), &unk_4F9D3C0);
  v94 = sub_14A4050(v19, a2);
  v20 = *(__int64 **)(a1 + 8);
  v21 = *v20;
  v22 = v20[1];
  if ( v21 == v22 )
LABEL_132:
    BUG();
  while ( *(_UNKNOWN **)v21 != &unk_4FBA0D1 )
  {
    v21 += 16;
    if ( v22 == v21 )
      goto LABEL_132;
  }
  v93 = *(_QWORD *)((*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v21 + 8) + 104LL))(
                      *(_QWORD *)(v21 + 8),
                      &unk_4FBA0D1)
                  + 160);
  v23 = *(__int64 **)(a1 + 8);
  v24 = *v23;
  v25 = v23[1];
  if ( v24 == v25 )
LABEL_131:
    BUG();
  while ( *(_UNKNOWN **)v24 != &unk_4FB9E3C )
  {
    v24 += 16;
    if ( v25 == v24 )
      goto LABEL_131;
  }
  v91 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v24 + 8) + 104LL))(*(_QWORD *)(v24 + 8), &unk_4FB9E3C);
  v92 = a1 + 160;
  v26 = sub_1AF0CE0(a2, 0, 0);
  v27 = *(_QWORD *)(a2 + 80);
  v108 = 0;
  v105 = v26;
  v109 = v27;
  v107 = a2 + 72;
  if ( v27 != a2 + 72 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v52 = v109;
        v53 = v109 - 24;
        v54 = v109 - 24;
        v109 = *(_QWORD *)(v109 + 8);
        v55 = sub_157EBA0(v54);
        if ( *(_BYTE *)(v55 + 16) == 25 )
          break;
LABEL_42:
        v27 = v109;
        if ( v107 == v109 )
          goto LABEL_56;
      }
      v56 = *(_QWORD **)(v52 + 24);
      if ( !v56 || (_QWORD *)v55 != v56 - 3 )
        break;
LABEL_22:
      if ( v108 )
      {
        v28 = *(_DWORD *)(v55 + 20) & 0xFFFFFFF;
        if ( v28 )
        {
LABEL_24:
          v101 = (__int64 **)(v55 - 24LL * v28);
          v29 = *v101;
          v30 = sub_157EBA0(v108);
          v35 = *(_DWORD *)(v30 + 20) & 0xFFFFFFF;
          v36 = 4 * v35;
          v99 = *(__int64 **)(v30 - 24 * v35);
          if ( v29 != v99 )
          {
            v37 = *(_QWORD *)(v108 + 48);
            if ( !v37 )
              BUG();
            v38 = v37 - 24;
            if ( *(_BYTE *)(v37 - 8) == 77 )
            {
              v102 = v37 - 24;
              v39 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
              if ( v39 != *(_DWORD *)(v38 + 56) )
              {
LABEL_28:
                v40 = (v39 + 1) & 0xFFFFFFF;
                v41 = v40 | *(_DWORD *)(v38 + 20) & 0xF0000000;
                *(_DWORD *)(v38 + 20) = v41;
                if ( (v41 & 0x40000000) != 0 )
                  v42 = *(_QWORD *)(v38 - 8);
                else
                  v42 = v102 - 24 * v40;
                v43 = (__int64 **)(v42 + 24LL * (unsigned int)(v40 - 1));
                if ( *v43 )
                {
                  v44 = v43[1];
                  v45 = (unsigned __int64)v43[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v45 = v44;
                  if ( v44 )
                    v44[2] = v44[2] & 3 | v45;
                }
                *v43 = v29;
                if ( v29 )
                {
                  v46 = v29[1];
                  v43[1] = (__int64 *)v46;
                  if ( v46 )
                    *(_QWORD *)(v46 + 16) = (unsigned __int64)(v43 + 1) | *(_QWORD *)(v46 + 16) & 3LL;
                  v43[2] = (__int64 *)((unsigned __int64)(v29 + 1) | (unsigned __int64)v43[2] & 3);
                  v29[1] = (__int64)v43;
                }
                v47 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
                v48 = (unsigned int)(v47 - 1);
                if ( (*(_BYTE *)(v38 + 23) & 0x40) != 0 )
                  v49 = *(_QWORD *)(v38 - 8);
                else
                  v49 = v102 - 24 * v47;
                *(_QWORD *)(v49 + 8 * v48 + 24LL * *(unsigned int *)(v38 + 56) + 8) = v53;
                v50 = (_QWORD *)sub_157EBA0(v53);
                sub_15F20C0(v50);
                v51 = sub_1648A60(56, 1u);
                if ( v51 )
                  sub_15F8590((__int64)v51, v108, v53);
                v10 = 1;
                goto LABEL_42;
              }
            }
            else
            {
              v63 = *(_QWORD *)(v108 + 8);
              if ( v63 )
              {
                while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v63) + 16) - 25) > 9u )
                {
                  v63 = *(_QWORD *)(v63 + 8);
                  if ( !v63 )
                    goto LABEL_99;
                }
                v112 = 1;
                v111 = 3;
                v110 = "merge";
                v96 = v55;
                v64 = 0;
                v65 = v63;
                while ( 1 )
                {
                  v65 = *(_QWORD *)(v65 + 8);
                  if ( !v65 )
                    break;
                  while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v65) + 16) - 25) <= 9u )
                  {
                    v65 = *(_QWORD *)(v65 + 8);
                    ++v64;
                    if ( !v65 )
                      goto LABEL_75;
                  }
                }
LABEL_75:
                v55 = v96;
                v97 = v64 + 1;
              }
              else
              {
LABEL_99:
                v112 = 1;
                v63 = 0;
                v110 = "merge";
                v111 = 3;
                v97 = 0;
              }
              v103 = **v101;
              v66 = sub_1648B60(64);
              v27 = v103;
              v67 = v66;
              if ( v66 )
              {
                v102 = v66;
                sub_15F1EA0(v66, v27, 53, 0, 0, v38);
                *(_DWORD *)(v67 + 56) = v97;
                sub_164B780(v67, (__int64 *)&v110);
                v27 = *(unsigned int *)(v67 + 56);
                sub_1648880(v67, v27, 1);
              }
              else
              {
                v102 = 0;
              }
              if ( v63 )
              {
                v68 = sub_1648700(v63);
                v72 = v99;
                v98 = v55;
                v95 = v53;
                v73 = v63;
                v27 = (__int64)(v99 + 1);
                v74 = (unsigned __int64)(v99 + 1);
LABEL_93:
                v82 = v68[5];
                v83 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
                if ( v83 == *(_DWORD *)(v67 + 56) )
                {
                  v100 = v82;
                  sub_15F55D0(v67, v27, v69, v70, v82, v71);
                  v82 = v100;
                  v83 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
                }
                v84 = (v83 + 1) & 0xFFFFFFF;
                v85 = v84 | *(_DWORD *)(v67 + 20) & 0xF0000000;
                *(_DWORD *)(v67 + 20) = v85;
                if ( (v85 & 0x40000000) != 0 )
                  v75 = *(_QWORD *)(v67 - 8);
                else
                  v75 = v102 - 24 * v84;
                v76 = (__int64 **)(v75 + 24LL * (unsigned int)(v84 - 1));
                if ( *v76 )
                {
                  v77 = v76[1];
                  v78 = (unsigned __int64)v76[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v78 = v77;
                  if ( v77 )
                    v77[2] = v77[2] & 3 | v78;
                }
                *v76 = v72;
                if ( v72 )
                {
                  v79 = v72[1];
                  v76[1] = (__int64 *)v79;
                  if ( v79 )
                    *(_QWORD *)(v79 + 16) = (unsigned __int64)(v76 + 1) | *(_QWORD *)(v79 + 16) & 3LL;
                  v76[2] = (__int64 *)(v74 | (unsigned __int64)v76[2] & 3);
                  v72[1] = (__int64)v76;
                }
                v80 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
                if ( (*(_BYTE *)(v67 + 23) & 0x40) != 0 )
                  v81 = *(_QWORD *)(v67 - 8);
                else
                  v81 = v102 - 24 * v80;
                *(_QWORD *)(v81 + 8LL * (unsigned int)(v80 - 1) + 24LL * *(unsigned int *)(v67 + 56) + 8) = v82;
                while ( 1 )
                {
                  v73 = *(_QWORD *)(v73 + 8);
                  if ( !v73 )
                    break;
                  v68 = sub_1648700(v73);
                  v27 = *((unsigned __int8 *)v68 + 16);
                  v69 = (unsigned int)(v27 - 25);
                  if ( (unsigned __int8)(v27 - 25) <= 9u )
                    goto LABEL_93;
                }
                v55 = v98;
                v53 = v95;
              }
              v86 = sub_157EBA0(v108);
              if ( (*(_BYTE *)(v86 + 23) & 0x40) != 0 )
                v87 = *(__int64 **)(v86 - 8);
              else
                v87 = (__int64 *)(v86 - 24LL * (*(_DWORD *)(v86 + 20) & 0xFFFFFFF));
              if ( *v87 )
              {
                v36 = v87[1];
                v88 = v87[2] & 0xFFFFFFFFFFFFFFFCLL;
                *(_QWORD *)v88 = v36;
                if ( v36 )
                {
                  v27 = *(_QWORD *)(v36 + 16) & 3LL;
                  *(_QWORD *)(v36 + 16) = v27 | v88;
                }
              }
              *v87 = v67;
              if ( v67 )
              {
                v89 = *(_QWORD *)(v67 + 8);
                v27 = v67 + 8;
                v87[1] = v89;
                if ( v89 )
                {
                  v36 = (unsigned __int64)(v87 + 1) | *(_QWORD *)(v89 + 16) & 3LL;
                  *(_QWORD *)(v89 + 16) = v36;
                }
                v87[2] = v27 | v87[2] & 3;
                *(_QWORD *)(v67 + 8) = v87;
              }
              v38 = v67;
              v90 = *(_DWORD *)(v55 + 20) & 0xFFFFFFF;
              v30 = 4 * v90;
              v29 = *(__int64 **)(v55 - 24 * v90);
              v39 = *(_DWORD *)(v67 + 20) & 0xFFFFFFF;
              if ( v39 != *(_DWORD *)(v67 + 56) )
                goto LABEL_28;
            }
            sub_15F55D0(v38, v27, v30, v36, v31, v32);
            v39 = *(_DWORD *)(v38 + 20) & 0xFFFFFFF;
            goto LABEL_28;
          }
        }
        v10 = 1;
        sub_164D160(v53, v108, a3, a4, a5, a6, v33, v34, a9, a10);
        sub_157F980(v53);
        goto LABEL_42;
      }
LABEL_55:
      v108 = v53;
      v27 = v109;
      if ( v107 == v109 )
      {
LABEL_56:
        v105 |= v10;
        goto LABEL_57;
      }
    }
    for ( i = (_QWORD *)(*(_QWORD *)(v55 + 24) & 0xFFFFFFFFFFFFFFF8LL); ; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( !i )
        BUG();
      v58 = *((_BYTE *)i - 8);
      if ( v58 != 78 )
        break;
      v62 = *(i - 6);
      if ( *(_BYTE *)(v62 + 16) || (*(_BYTE *)(v62 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v62 + 36) - 35) > 3 )
        goto LABEL_42;
      if ( v56 == i )
        goto LABEL_22;
    }
    if ( v56 != i )
      goto LABEL_42;
    if ( v58 != 77 )
      goto LABEL_42;
    v59 = *(_DWORD *)(v55 + 20) & 0xFFFFFFF;
    if ( !v59 )
      goto LABEL_42;
    v27 = 4LL * v59;
    v60 = *(_QWORD **)(v55 - 24LL * v59);
    if ( v60 != i - 3 || !v60 )
      goto LABEL_42;
    v28 = *(_DWORD *)(v55 + 20) & 0xFFFFFFF;
    if ( !v108 )
      goto LABEL_55;
    goto LABEL_24;
  }
LABEL_57:
  if ( !*(_BYTE *)(a1 + 168) )
  {
    v105 |= sub_1A61790(a2, v94, v93, v91, v92);
    if ( !v105 )
      return 0;
    if ( (unsigned __int8)sub_1AF0CE0(a2, 0, 0) )
    {
      while ( (unsigned __int8)sub_1A61790(a2, v94, v93, v91, v92) )
        sub_1AF0CE0(a2, 0, 0);
    }
  }
  return v105;
}
