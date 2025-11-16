// Function: sub_6C5750
// Address: 0x6c5750
//
__int64 __fastcall sub_6C5750(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        int a5,
        int a6,
        int a7,
        int a8,
        __int64 a9,
        unsigned int a10,
        __int64 a11,
        __int64 a12,
        _DWORD *a13,
        int *a14,
        _DWORD *a15,
        _DWORD *a16,
        _DWORD *a17,
        int a18,
        __int64 *a19,
        __int64 *a20,
        _QWORD *a21)
{
  __int64 v21; // r13
  int v24; // r15d
  __int16 v25; // dx
  char v26; // di
  __int64 v27; // r14
  __int64 i; // rax
  __int64 v29; // rax
  __int64 v30; // r11
  __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // rdi
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 result; // rax
  __int64 j; // r11
  __int64 v38; // r15
  _QWORD *v39; // r8
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // r10
  __int64 v43; // rax
  __int64 v44; // rbx
  __int64 v45; // rdi
  __int64 v46; // r9
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 k; // rdx
  __int64 v50; // rdi
  int v51; // ebx
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rsi
  __int64 v57; // rdi
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // rax
  __int64 v66; // rdx
  char v67; // al
  __int64 m; // rdi
  int v69; // eax
  __int64 n; // rdx
  __int64 ii; // rdi
  int v72; // eax
  int v73; // eax
  __int64 v74; // rax
  int v75; // eax
  __int64 v76; // [rsp-8h] [rbp-F8h]
  __int64 v77; // [rsp+8h] [rbp-E8h]
  __int64 v78; // [rsp+8h] [rbp-E8h]
  _QWORD *v79; // [rsp+10h] [rbp-E0h]
  __int64 v80; // [rsp+10h] [rbp-E0h]
  __int64 v81; // [rsp+10h] [rbp-E0h]
  __int64 v82; // [rsp+10h] [rbp-E0h]
  __int64 v83; // [rsp+10h] [rbp-E0h]
  __int64 v84; // [rsp+10h] [rbp-E0h]
  __int64 v85; // [rsp+10h] [rbp-E0h]
  __int64 v86; // [rsp+18h] [rbp-D8h]
  _QWORD *v87; // [rsp+20h] [rbp-D0h]
  __int64 v88; // [rsp+20h] [rbp-D0h]
  unsigned int v89; // [rsp+20h] [rbp-D0h]
  __int64 v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+20h] [rbp-D0h]
  __int64 v92; // [rsp+20h] [rbp-D0h]
  __int64 v93; // [rsp+28h] [rbp-C8h]
  int v94; // [rsp+28h] [rbp-C8h]
  __int64 v95; // [rsp+28h] [rbp-C8h]
  __int64 v96; // [rsp+28h] [rbp-C8h]
  int v97; // [rsp+28h] [rbp-C8h]
  __int64 v98; // [rsp+28h] [rbp-C8h]
  __int64 v99; // [rsp+30h] [rbp-C0h]
  int v100; // [rsp+30h] [rbp-C0h]
  __int64 v102; // [rsp+40h] [rbp-B0h]
  int v103; // [rsp+40h] [rbp-B0h]
  unsigned int v105; // [rsp+50h] [rbp-A0h]
  __int64 v106; // [rsp+50h] [rbp-A0h]
  bool v108; // [rsp+5Dh] [rbp-93h]
  __int16 v109; // [rsp+5Eh] [rbp-92h]
  unsigned int v110; // [rsp+60h] [rbp-90h] BYREF
  int v111; // [rsp+64h] [rbp-8Ch] BYREF
  __int64 v112; // [rsp+68h] [rbp-88h] BYREF
  __int64 *v113; // [rsp+70h] [rbp-80h] BYREF
  __int64 v114; // [rsp+78h] [rbp-78h] BYREF
  __int64 v115; // [rsp+80h] [rbp-70h] BYREF
  __int64 v116; // [rsp+88h] [rbp-68h] BYREF
  _QWORD v117[5]; // [rsp+90h] [rbp-60h] BYREF
  char v118; // [rsp+B8h] [rbp-38h]

  v21 = a2;
  v24 = a8;
  v110 = 0;
  v25 = *(_WORD *)(qword_4D03C50 + 20LL);
  v26 = *(_BYTE *)(qword_4D03C50 + 21LL);
  v112 = 0;
  v113 = 0;
  v109 = v25;
  v25 &= 0xFBF7u;
  HIBYTE(v25) |= 4u;
  v115 = 0;
  v116 = 0;
  v111 = 0;
  v108 = (v26 & 4) != 0;
  *(_WORD *)(qword_4D03C50 + 20LL) = v25;
  if ( a13 )
    *a13 = 0;
  if ( a14 )
    *a14 = 0;
  if ( a15 )
    *a15 = 0;
  if ( a16 )
    *a16 = 0;
  if ( a17 )
    *a17 = 0;
  v27 = *(_QWORD *)(a1 + 64);
  for ( i = v27; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v29 = *(_QWORD *)i;
  if ( !a3 )
    a3 = *(_QWORD *)(a1 + 64);
  v99 = *(_QWORD *)(v29 + 96);
  if ( *(_BYTE *)(a1 + 80) != 10 )
  {
    sub_6C0910(0, 0, 1u, &v114, 1, 0, 0, a7, a9, a10, a11, &v112, 0, 0, a21);
    v30 = v112;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)a2;
    v105 = dword_4D04230;
    if ( !dword_4D04230 )
      goto LABEL_17;
LABEL_34:
    v105 = v30 == 0;
    goto LABEL_17;
  }
  for ( j = *(_QWORD *)(*(_QWORD *)(a1 + 88) + 152LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v102 = j;
  v106 = *(_QWORD *)(a1 + 88);
  v94 = sub_72F500(v106, 0, 0, 1, 0);
  if ( !v94 )
  {
    sub_6C0910(v102, v106, 1u, &v114, 0, 0, 0, a7, a9, a10, a11, &v112, 0, 0, a21);
    v32 = v112;
    *(_QWORD *)dword_4F07508 = *(_QWORD *)a2;
    v105 = dword_4D04230;
    if ( dword_4D04230 )
      v105 = v114 == 0;
    goto LABEL_36;
  }
  sub_6C0910(0, 0, 1u, &v114, 1, 0, 0, a7, a9, a10, a11, &v112, 0, 0, a21);
  v30 = v112;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)a2;
  v105 = dword_4D04230;
  if ( dword_4D04230 )
    goto LABEL_34;
LABEL_17:
  v93 = v30;
  if ( (a8 & 0x400) == 0 )
    v24 = a8 | 0x400000;
  v31 = sub_84AC10(
          a1,
          0,
          0,
          1,
          0,
          v30,
          a12,
          v24,
          0,
          0,
          0,
          2,
          a2,
          0,
          0,
          (__int64)&v111,
          (__int64)&v110,
          0,
          0,
          (__int64)&v113);
  v32 = v93;
  a1 = v31;
  if ( !v31 )
  {
    sub_84A700(0, v110, 0, 1, 0, v93, (__int64)v113, (__int64)&v114);
    v33 = v115;
    a2 = v76;
    if ( v115 || !v110 )
      goto LABEL_21;
    v38 = 0;
    if ( a7 )
      goto LABEL_75;
    goto LABEL_46;
  }
  v94 = 1;
LABEL_36:
  if ( v111 )
    v32 = a12;
  v38 = *(_QWORD *)(a1 + 88);
  if ( a14 )
    *a14 = *(_BYTE *)(v38 + 193) >> 7;
  if ( (*(_BYTE *)(v38 + 194) & 2) == 0 || v32 )
  {
    if ( a6 )
    {
      v87 = (_QWORD *)v32;
      v40 = sub_72F500(v38, 0, 0, 1, 0);
      LODWORD(v32) = (_DWORD)v87;
      if ( v40 )
      {
        if ( v87 && !*v87 && (*(_BYTE *)(*v113 + 64) & 0x88) == 0 )
        {
          v79 = v87;
          v88 = *v113;
          v41 = sub_8D46C0(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)(v38 + 152) + 168LL) + 8LL));
          v32 = (__int64)v79;
          v42 = v88;
          v86 = v41;
          if ( *((_BYTE *)v79 + 8)
            || (v65 = v79[3], *(_BYTE *)(v65 + 24) == 5) && (v32 = *(_QWORD *)(v65 + 152), *(_BYTE *)(v32 + 8)) )
          {
LABEL_57:
            v80 = v32;
            v95 = v42;
            sub_83EB20(v86, v38, a2);
            v42 = v95;
            v89 = 0;
            v103 = 0;
            v32 = v80;
LABEL_58:
            if ( a15 )
              *a15 = 1;
            *(_BYTE *)(qword_4D03C50 + 21LL) = (4 * v108) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB;
            if ( *(_BYTE *)(v32 + 8) == 1 )
            {
              v97 = v32;
              sub_6E6990(v117);
              a2 = v86;
              v51 = 0;
              v118 |= 8u;
              sub_839D30(v97, v86, 0, 1, 0, 0, a5, 1, 0, 0, (__int64)v117, 0);
              v115 = v117[1];
            }
            else
            {
              v43 = *(_QWORD *)(v32 + 24);
              v44 = v43 + 8;
              if ( !*(_QWORD *)(v42 + 48) && (*(_WORD *)(v42 + 64) & 0x101) == 0 )
                *(_BYTE *)(v42 + 64) |= 0x10u;
              if ( v103 )
                *(_BYTE *)(v42 + 64) &= ~4u;
              v45 = v43 + 8;
              v96 = v43;
              sub_8449E0(v43 + 8, v27, v42 + 48, 0, 0);
              if ( !v89 )
                goto LABEL_100;
              v47 = *(unsigned __int8 *)(v96 + 24);
              if ( !(_BYTE)v47 )
                goto LABEL_100;
              v48 = *(_QWORD *)(v96 + 8);
              for ( k = *(unsigned __int8 *)(v48 + 140); (_BYTE)k == 12; k = *(unsigned __int8 *)(v48 + 140) )
                v48 = *(_QWORD *)(v48 + 160);
              if ( (_BYTE)k )
              {
                if ( (_BYTE)v47 == 2 )
                {
                  v117[0] = sub_724DC0(v45, v27, k, v47, v89, v46);
                  v55 = sub_725A70(2);
                  v56 = v117[0];
                  v57 = v44;
                  v51 = 0;
                  v115 = v55;
                  sub_6F4950(v57, v117[0], v58, v59, v60, v61);
                  a2 = sub_724E50(v117, v56, v62, v63, v64);
                  sub_72F900(v115, a2);
                }
                else
                {
                  v50 = v44;
                  v51 = 0;
                  sub_831A40(v50, a5 == 0, &v116, &v115);
                  a2 = 0;
                  sub_83EB20(v86, 0, v21);
                }
              }
              else
              {
LABEL_100:
                v54 = v44;
                a2 = 0;
                v51 = 0;
                v114 = sub_6F6F40(v54, 0);
              }
            }
            goto LABEL_84;
          }
          v66 = *(_QWORD *)(v88 + 48);
          if ( v66 )
            goto LABEL_113;
          if ( (*(_WORD *)(v88 + 64) & 0x101) == 0 )
          {
            for ( m = *(_QWORD *)(*(_QWORD *)(v32 + 24) + 8LL); *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
              ;
            v82 = v32;
            v69 = sub_8D0520(m, v27);
            v42 = v88;
            v32 = v82;
            if ( v69 )
            {
              v74 = *(_QWORD *)(v82 + 24);
              v85 = v88;
              v92 = v32;
              v75 = sub_831A40(v74 + 8, a5 == 0, &v116, &v115);
              v32 = v92;
              v42 = v85;
              if ( v75 )
              {
                if ( !a12 )
                  *(_BYTE *)(v115 + 50) &= ~0x10u;
                goto LABEL_57;
              }
            }
            v66 = *(_QWORD *)(v42 + 48);
            if ( v66 )
            {
LABEL_113:
              v67 = *(_BYTE *)(v66 + 174);
              if ( v67 == 1 )
              {
                v78 = v32;
                v84 = v42;
                v91 = v66;
                v73 = sub_8D0520(*(_QWORD *)(*(_QWORD *)(v66 + 40) + 32LL), v27);
                v66 = v91;
                v42 = v84;
                v32 = v78;
                if ( v73 )
                {
LABEL_131:
                  *(_BYTE *)(v42 + 64) &= ~0x10u;
                  v89 = 1;
                  v103 = 0;
                  goto LABEL_58;
                }
                v67 = *(_BYTE *)(v91 + 174);
              }
              if ( v67 == 3 )
              {
                for ( n = *(_QWORD *)(v66 + 152); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
                  ;
                for ( ii = *(_QWORD *)(n + 160); *(_BYTE *)(ii + 140) == 12; ii = *(_QWORD *)(ii + 160) )
                  ;
                v77 = v32;
                v90 = v42;
                v83 = n;
                v72 = sub_8D0520(ii, v27);
                v42 = v90;
                v32 = v77;
                if ( v72 )
                {
                  if ( (*(_BYTE *)(*(_QWORD *)(v83 + 168) + 16LL) & 0x20) != 0 )
                    goto LABEL_131;
                }
              }
            }
          }
          if ( (*(_BYTE *)(v38 + 194) & 4) != 0 )
          {
            v81 = v32;
            v98 = v42;
            sub_6E6080(a1, a2, a3, 0, 0, 0);
            v42 = v98;
            v89 = 0;
            v103 = 1;
            v32 = v81;
            goto LABEL_58;
          }
        }
      }
    }
    v100 = v32;
    sub_6E6130(a1, a2, a3, 0);
    if ( v94 )
    {
      a2 = v110;
      sub_84A700(a1, v110, 0, 1, 0, v100, (__int64)v113, (__int64)&v114);
      v33 = v115;
      if ( v115 )
        goto LABEL_21;
    }
    else
    {
      v33 = v115;
      if ( v115 )
        goto LABEL_21;
    }
    if ( a7 )
    {
LABEL_75:
      v52 = sub_6EAFA0(5);
      *(_QWORD *)(v52 + 56) = v38;
      v33 = v52;
      v115 = v52;
      *(_QWORD *)(v52 + 64) = v114;
      *(_BYTE *)(v52 + 72) = (4 * (v105 & 1)) | *(_BYTE *)(v52 + 72) & 0xFB;
      goto LABEL_76;
    }
    goto LABEL_46;
  }
  sub_6E6080(a1, a2, a3, 0, 0, 0);
  if ( a15 )
    *a15 = 1;
  v51 = 1;
  *(_BYTE *)(qword_4D03C50 + 21LL) = (4 * ((v26 & 4) != 0)) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB;
  v103 = 0;
LABEL_84:
  sub_82D8A0(v113);
  v53 = unk_4D03C20;
  if ( unk_4D03C20 )
  {
    a2 = 0;
    unk_4D03C20 = 0;
    sub_6E5270(v53, 0, &unk_4D03C28, &unk_4D03C30);
  }
  v33 = v115;
  if ( v115 )
    goto LABEL_21;
  if ( a7 )
    goto LABEL_75;
  if ( v103 )
  {
    v33 = sub_6EAFA0(3);
    v115 = v33;
    *(_QWORD *)(v33 + 56) = v114;
LABEL_76:
    if ( !a5 )
      goto LABEL_21;
    goto LABEL_77;
  }
  if ( v51 )
  {
    if ( v105 )
    {
      v115 = sub_6EAFA0(1);
      v33 = v115;
    }
    else
    {
      if ( a13 && (!a5 || !*(_QWORD *)(v99 + 24) || (*(_BYTE *)(v99 + 177) & 2) != 0) )
      {
        v33 = 0;
        *a13 = 1;
        goto LABEL_21;
      }
      v115 = sub_6EAFA0(0);
      v33 = v115;
    }
    goto LABEL_96;
  }
LABEL_46:
  *(_BYTE *)(qword_4D03C50 + 21LL) = (4 * v108) | *(_BYTE *)(qword_4D03C50 + 21LL) & 0xFB;
  a2 = v114;
  v115 = sub_6F5430(v38, v114, a4, 0, 0, 0, v105, a12 != 0, a6, a6, v21);
  if ( (unsigned int)sub_730800(v115) )
  {
    v115 = 0;
    v33 = 0;
    goto LABEL_21;
  }
  v33 = v115;
LABEL_96:
  if ( a5 && v33 )
  {
LABEL_77:
    a2 = v27;
    sub_6EB360(v33, v27, a3, v21);
    v33 = v115;
  }
LABEL_21:
  *a19 = v33;
  if ( a20 )
  {
    if ( a13 && *a13 )
    {
      v116 = 0;
      v34 = 0;
    }
    else
    {
      v39 = (_QWORD *)v116;
      if ( v116 )
      {
        if ( (*(_BYTE *)(v116 + 25) & 3) != 0 )
        {
          v116 = sub_731370(v116, a2);
          v39 = (_QWORD *)v116;
        }
        *v39 = a4;
        sub_6EB510(v116, a2);
        v34 = v116;
      }
      else
      {
        if ( v33 )
          v34 = sub_6EC670(a4, v33, 0, 0);
        else
          v34 = sub_7305B0(0, a2);
        v116 = v34;
      }
    }
    *a20 = v34;
  }
  if ( v112 && v112 != a11 )
    sub_6E1990(v112);
  v35 = *(unsigned __int16 *)(qword_4D03C50 + 20LL);
  LOWORD(v35) = v35 & 0xFBF7;
  result = v109 & 0x408 | v35;
  *(_WORD *)(qword_4D03C50 + 20LL) = result;
  return result;
}
