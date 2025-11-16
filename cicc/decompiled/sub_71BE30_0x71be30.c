// Function: sub_71BE30
// Address: 0x71be30
//
__int64 __fastcall sub_71BE30(__int64 a1)
{
  __int64 result; // rax
  _BYTE *v2; // r15
  __int64 v3; // rsi
  __int64 i; // r12
  __int64 v5; // rcx
  __int64 v6; // rax
  char v7; // al
  __int64 v8; // r14
  _QWORD *v9; // r13
  int v10; // ebx
  __int64 v11; // r15
  unsigned int v12; // r8d
  __int64 v13; // r13
  __int64 j; // rax
  _QWORD *k; // r14
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // r8
  _BYTE *v24; // rax
  _BYTE *m; // rax
  __int64 v26; // r13
  __int64 n; // rax
  __int64 *v28; // rax
  __int64 v29; // rbx
  int v30; // eax
  __int64 v31; // r14
  __int64 v32; // rcx
  __int64 v33; // rbx
  __int64 v34; // rax
  __int64 *v35; // r12
  char v36; // al
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // r13
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdi
  _QWORD *v45; // r15
  __int64 ii; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 jj; // r15
  __int64 v50; // rbx
  __int64 v51; // r13
  char kk; // al
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 *v55; // r14
  __int64 v56; // rax
  __int64 v57; // rdx
  _QWORD *v58; // r9
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rbx
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  _DWORD *v70; // rax
  __int64 mm; // rcx
  __int64 v72; // rax
  __int64 v73; // rdx
  __int64 v74; // r14
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // r9
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // rax
  _BYTE *v82; // rcx
  __int64 v83; // rax
  __int64 nn; // r9
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 *v88; // r14
  __int64 v89; // rax
  __int64 v90; // rdi
  __int64 v91; // rsi
  __int64 v92; // rax
  __int64 v93; // [rsp+8h] [rbp-108h]
  __int64 v94; // [rsp+8h] [rbp-108h]
  _BYTE *v95; // [rsp+10h] [rbp-100h]
  __int64 v96; // [rsp+10h] [rbp-100h]
  __int64 v97; // [rsp+18h] [rbp-F8h]
  __int64 v98; // [rsp+18h] [rbp-F8h]
  __int64 v99; // [rsp+20h] [rbp-F0h]
  __int64 v100; // [rsp+20h] [rbp-F0h]
  __int64 v101; // [rsp+20h] [rbp-F0h]
  __int64 v102; // [rsp+20h] [rbp-F0h]
  _QWORD *v103; // [rsp+20h] [rbp-F0h]
  _QWORD *v104; // [rsp+20h] [rbp-F0h]
  __int64 v106; // [rsp+28h] [rbp-E8h]
  _BYTE *v107; // [rsp+28h] [rbp-E8h]
  __int64 v108; // [rsp+28h] [rbp-E8h]
  __int64 *v109; // [rsp+30h] [rbp-E0h]
  unsigned __int16 v110; // [rsp+3Eh] [rbp-D2h]
  int v111; // [rsp+40h] [rbp-D0h]
  int v112; // [rsp+44h] [rbp-CCh]
  __int64 v113; // [rsp+48h] [rbp-C8h]
  __int64 v114; // [rsp+50h] [rbp-C0h]
  unsigned int v115; // [rsp+50h] [rbp-C0h]
  _BYTE *v116; // [rsp+60h] [rbp-B0h]
  char *v117; // [rsp+60h] [rbp-B0h]
  __int64 v118; // [rsp+60h] [rbp-B0h]
  __int64 *v119; // [rsp+68h] [rbp-A8h]
  unsigned int v120[3]; // [rsp+74h] [rbp-9Ch] BYREF
  char v121; // [rsp+80h] [rbp-90h] BYREF
  char *v122; // [rsp+90h] [rbp-80h]

  result = *(_QWORD *)(a1 + 40);
  v2 = *(_BYTE **)(result + 32);
  if ( (v2[177] & 0x20) != 0 )
    return result;
  v3 = *(_QWORD *)(a1 + 152);
  for ( i = a1; *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
    ;
  v119 = (__int64 *)sub_71B6A0(a1, v3, (__int64)v2, v120);
  if ( (*(_DWORD *)(a1 + 192) & 0x400200) == (_DWORD)&dword_400200 )
  {
    v3 = (__int64)v2;
    if ( !(unsigned int)sub_6009B0(a1, (__int64)v2, 0) )
      *(_BYTE *)(a1 + 193) &= 0xFCu;
  }
  v6 = *(_QWORD *)(a1 + 152);
  if ( *(_BYTE *)(v6 + 140) == 7 )
  {
    v24 = *(_BYTE **)(*(_QWORD *)(v6 + 168) + 56LL);
    if ( v24 )
    {
      if ( (*v24 & 2) != 0 )
        sub_5F80E0(a1);
    }
  }
  v7 = *(_BYTE *)(a1 + 174);
  if ( v7 == 1 )
  {
    v13 = v119[4];
    for ( j = *(_QWORD *)(v13 + 152); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    for ( k = **(_QWORD ***)(j + 168); k; k = (_QWORD *)*k )
    {
      v16 = k[1];
      v17 = sub_735FB0(v16, 3, 0xFFFFFFFFLL, v5);
      *(_BYTE *)(v17 + 169) |= 0x80u;
      v18 = v17;
      *(_BYTE *)(v17 + 89) |= 1u;
      *(_QWORD *)(v17 + 256) = v16;
      sub_72FBE0(v17);
      *(_QWORD *)(v18 + 128) = k;
    }
    if ( (*(_BYTE *)(v13 + 194) & 0x40) != 0 )
      v19 = sub_640AA0(v13);
    else
      v19 = sub_63CAE0(v13, 0, 0, v5);
    v119[6] = v19;
    v20 = sub_726B30(11);
    v119[10] = v20;
    v21 = v20;
    goto LABEL_24;
  }
  if ( v7 == 2 )
  {
    v13 = v119[4];
    if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v13 + 40) + 32LL) + 140LL) != 11 )
      v119[6] = (__int64)sub_63FE70(v119[4]);
    v21 = sub_726B30(11);
    v119[10] = v21;
LABEL_24:
    *(_QWORD *)(v21 + 72) = sub_726B30(8);
    sub_71BD50(v13);
    return sub_71B580(i, (__int64)v119, v120, v22, v23);
  }
  if ( (v2[176] & 2) != 0 )
    goto LABEL_97;
  for ( m = v2; m[140] == 12; m = (_BYTE *)*((_QWORD *)m + 20) )
    ;
  if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)m + 96LL) + 178LL) & 0x20) != 0 )
  {
LABEL_97:
    v8 = **(_QWORD **)(*(_QWORD *)v2 + 96LL);
    if ( v8 )
    {
      v116 = v2;
      v9 = 0;
      v10 = 0;
      while ( 1 )
      {
        if ( *(_BYTE *)(v8 + 80) == 8 )
        {
          v11 = *(_QWORD *)(*(_QWORD *)(v8 + 88) + 120LL);
          if ( (unsigned int)sub_8D32E0(v11) )
          {
            v12 = 420;
            goto LABEL_76;
          }
          if ( (*(_BYTE *)(v11 + 140) & 0xFB) == 8 )
          {
            v3 = dword_4F077C4 != 2;
            if ( (sub_8D4C10(v11, v3) & 1) != 0 )
            {
              v12 = 419;
LABEL_76:
              if ( !v10 )
              {
                v115 = v12;
                v70 = sub_67D9D0(0x18Du, (_DWORD *)v116 + 16);
                v12 = v115;
                v9 = v70;
              }
              v3 = v12;
              v10 = 1;
              sub_67E1D0(v9, v12, v8);
            }
          }
        }
        v8 = *(_QWORD *)(v8 + 16);
        if ( !v8 )
        {
          if ( v10 )
            sub_685910((__int64)v9, (FILE *)v3);
          break;
        }
      }
    }
  }
  v111 = dword_4F07508[0];
  v110 = dword_4F07508[1];
  v26 = v119[4];
  for ( n = *(_QWORD *)(v26 + 152); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
    ;
  v28 = *(__int64 **)(n + 168);
  v29 = *v28;
  v30 = sub_8D3110(*(_QWORD *)(*v28 + 8));
  v31 = *(_QWORD *)(v29 + 8);
  v112 = v30;
  v113 = sub_735FB0(v31, 3, 0xFFFFFFFFLL, v32);
  *(_BYTE *)(v113 + 169) |= 0x80u;
  *(_BYTE *)(v113 + 89) |= 1u;
  *(_QWORD *)(v113 + 256) = v31;
  sub_72FBE0(v113);
  *(_QWORD *)(v113 + 128) = v29;
  v33 = sub_8D46C0(*(_QWORD *)(v119[8] + 120));
  *(_QWORD *)dword_4F07508 = *(_QWORD *)(v33 + 64);
  v34 = sub_726B30(11);
  v122 = 0;
  v114 = v34;
  v119[10] = v34;
  if ( (*(_BYTE *)(v26 + 194) & 4) != 0 )
  {
    v61 = sub_71AD70(v113);
    v62 = sub_731370(v61, 3);
    v63 = sub_73E870();
    v64 = sub_73DCD0(v63);
    v117 = (char *)sub_73E690(v64, v62);
    v122 = v117;
    *((_QWORD *)v117 + 3) = v114;
    goto LABEL_72;
  }
  v109 = (__int64 *)(v33 + 64);
  v117 = &v121;
  if ( !**(_QWORD **)(v33 + 168) )
    goto LABEL_51;
  v35 = **(__int64 ***)(v33 + 168);
  do
  {
    while ( 1 )
    {
      v36 = *((_BYTE *)v35 + 96);
      if ( (v36 & 1) == 0 || (v36 & 2) != 0 && (unsigned int)sub_8E35E0(v35, v33) )
        goto LABEL_38;
      v37 = sub_73E870();
      v38 = sub_73E4A0(v37, v35);
      v39 = (__int64 *)sub_73DCD0(v38);
      v40 = sub_71AD70(v113);
      v41 = sub_73E1B0(v40, v35);
      v42 = sub_73E4A0(v41, v35);
      v43 = sub_73DCD0(v42);
      v44 = v35[5];
      v45 = (_QWORD *)v43;
      for ( ii = v44; *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
        ;
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)ii + 96LL) + 177LL) & 0x20) != 0 )
        break;
      if ( v112 )
      {
        v69 = sub_731380(v45);
        v44 = v35[5];
        v45 = (_QWORD *)v69;
      }
      v47 = sub_697CE0(v44, v45, v39, (_DWORD *)v35 + 18);
      if ( v47 )
      {
        v48 = sub_71AEA0((__int64)v45, v39, v47, v109);
        goto LABEL_49;
      }
LABEL_38:
      v35 = (__int64 *)*v35;
      if ( !v35 )
        goto LABEL_50;
    }
    v68 = sub_731370(v45, v35);
    v48 = sub_73E690(v39, v68);
LABEL_49:
    *((_QWORD *)v117 + 2) = v48;
    *(_QWORD *)(v48 + 24) = v114;
    v35 = (__int64 *)*v35;
    v117 = (char *)v48;
  }
  while ( v35 );
LABEL_50:
  i = a1;
LABEL_51:
  for ( jj = **(_QWORD **)(*(_QWORD *)v33 + 96LL); jj; jj = *(_QWORD *)(jj + 16) )
  {
    if ( *(_BYTE *)(jj + 80) != 8 )
      continue;
    v50 = *(_QWORD *)(jj + 88);
    v51 = *(_QWORD *)(v50 + 120);
    for ( kk = *(_BYTE *)(v51 + 140); kk == 12; kk = *(_BYTE *)(v51 + 140) )
      v51 = *(_QWORD *)(v51 + 160);
    if ( kk == 8 && (sub_8D4C10(v51, dword_4F077C4 != 2) & 1) != 0 || (unsigned int)sub_8D32E0(v51) )
      continue;
    if ( (unsigned int)sub_8D3410(v51) )
    {
      for ( mm = sub_8D40F0(v51); *(_BYTE *)(mm + 140) == 12; mm = *(_QWORD *)(mm + 160) )
        ;
      v107 = (_BYTE *)mm;
      v72 = sub_73E870();
      v74 = sub_73E470(v72, v50, v73);
      v75 = sub_71AD70(v113);
      v77 = sub_73E470(v75, v50, v76);
      if ( (unsigned __int8)(v107[140] - 9) <= 2u
        && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v107 + 96LL) + 177LL) & 0x20) == 0 )
      {
        v93 = v77;
        v95 = v107;
        v97 = sub_72BA30(byte_4F06A51[0]);
        v108 = sub_736020(v97, 0);
        v100 = sub_731250(v108);
        v80 = sub_73A830(0, byte_4F06A51[0]);
        v101 = sub_73E690(v100, v80);
        *((_QWORD *)v117 + 2) = v101;
        *(_QWORD *)(v101 + 24) = v114;
        v81 = sub_731250(v108);
        v118 = sub_73DBF0(37, v97, v81);
        v82 = v95;
        v83 = v51;
        for ( nn = v93; *(_BYTE *)(v83 + 140) == 12; v83 = *(_QWORD *)(v83 + 160) )
          ;
        v94 = v101;
        v96 = nn;
        v98 = (__int64)v82;
        *(_QWORD *)(v118 + 16) = sub_73A8E0(*(_QWORD *)(v83 + 128) / *((_QWORD *)v82 + 16), byte_4F06A51[0]);
        v85 = sub_6EFF80();
        v102 = sub_73DBF0(61, v85, v118);
        v117 = (char *)sub_726B30(12);
        *(_QWORD *)(v94 + 16) = v117;
        *((_QWORD *)v117 + 3) = v114;
        *((_QWORD *)v117 + 6) = v102;
        v103 = sub_71AE50(v96);
        v103[2] = sub_73E830(v108);
        v86 = sub_8D46C0(*v103);
        v87 = sub_73DBF0(92, v86, v103);
        *(_BYTE *)(v87 + 25) |= 1u;
        v104 = (_QWORD *)v87;
        v88 = sub_71AE50(v74);
        v89 = sub_73E830(v108);
        v90 = *v88;
        v88[2] = v89;
        v91 = sub_8D46C0(v90);
        v92 = sub_73DBF0(92, v91, v88);
        *(_BYTE *)(v92 + 25) |= 1u;
        v55 = (__int64 *)v92;
        v106 = v51;
        v58 = v104;
        v51 = v98;
        goto LABEL_64;
      }
      v78 = sub_73E6E0(v74, v77);
    }
    else
    {
      v53 = sub_73E870();
      v55 = (__int64 *)sub_73E470(v53, v50, v54);
      v56 = sub_71AD70(v113);
      v58 = (_QWORD *)sub_73E470(v56, v50, v57);
      if ( (unsigned __int8)(*(_BYTE *)(v51 + 140) - 9) <= 2u
        && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v51 + 96LL) + 177LL) & 0x20) == 0 )
      {
        v106 = 0;
LABEL_64:
        if ( v112 )
          v58 = (_QWORD *)sub_731380(v58);
        v99 = (__int64)v58;
        v59 = sub_697CE0(v51, v58, v55, (_DWORD *)(v50 + 64));
        if ( v59 )
        {
          v60 = sub_71AEA0(v99, v55, v59, v109);
          if ( v106 )
          {
            *((_QWORD *)v117 + 9) = v60;
            *(_QWORD *)(v60 + 24) = v117;
          }
          else
          {
            *((_QWORD *)v117 + 2) = v60;
            *(_QWORD *)(v60 + 24) = v114;
            v117 = (char *)v60;
          }
        }
        continue;
      }
      v79 = sub_731370(v58, v50);
      v78 = sub_73E690(v55, v79);
    }
    *((_QWORD *)v117 + 2) = v78;
    *(_QWORD *)(v78 + 24) = v114;
    v117 = (char *)v78;
  }
LABEL_72:
  v65 = sub_726B30(8);
  *((_QWORD *)v117 + 2) = v65;
  *(_QWORD *)(v65 + 24) = v114;
  v66 = sub_73E870();
  v67 = sub_73DCD0(v66);
  *(_QWORD *)(v65 + 48) = sub_73E250(v67);
  *(_QWORD *)(v114 + 72) = v122;
  dword_4F07508[0] = v111;
  v22 = v110;
  LOWORD(dword_4F07508[1]) = v110;
  return sub_71B580(i, (__int64)v119, v120, v22, v23);
}
