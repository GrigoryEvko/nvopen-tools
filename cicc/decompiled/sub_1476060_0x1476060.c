// Function: sub_1476060
// Address: 0x1476060
//
__int64 *__fastcall sub_1476060(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r15
  __int64 v4; // r12
  __int64 *v6; // r12
  __int64 v8; // rax
  unsigned int v9; // ebx
  unsigned int v10; // r13d
  __int16 v11; // ax
  __int64 v12; // rbx
  unsigned int v13; // ebx
  __int64 v14; // rax
  unsigned int v15; // eax
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // rbx
  unsigned int v21; // ebx
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  unsigned int v29; // ebx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r15
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  unsigned int v37; // ebx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int16 v47; // r13
  __int64 *v48; // rax
  char v49; // r13
  __int64 v50; // rbx
  __int64 v51; // r14
  char v52; // r12
  char v53; // dl
  unsigned int v54; // eax
  unsigned int v55; // eax
  __int64 v56; // rdi
  __int64 v57; // rsi
  __int64 v58; // rax
  unsigned int v59; // eax
  __int64 v60; // r13
  unsigned int v61; // eax
  unsigned int v62; // eax
  unsigned int v63; // r13d
  unsigned int v64; // r13d
  __int64 v65; // rcx
  __int64 v66; // r8
  __int64 *v67; // rcx
  int v68; // eax
  __int64 *v69; // r12
  __int64 v70; // rsi
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // r8
  _QWORD *v75; // rdx
  _QWORD *v76; // rax
  _QWORD *v77; // rcx
  __int64 v78; // rdx
  _QWORD *v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // rcx
  __int64 v82; // rcx
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 *v86; // [rsp+10h] [rbp-170h]
  char v87; // [rsp+20h] [rbp-160h]
  __int64 v88; // [rsp+20h] [rbp-160h]
  unsigned int v89; // [rsp+28h] [rbp-158h]
  int v90; // [rsp+30h] [rbp-150h]
  int v91; // [rsp+30h] [rbp-150h]
  int v92; // [rsp+30h] [rbp-150h]
  int v93; // [rsp+30h] [rbp-150h]
  __int64 v94; // [rsp+30h] [rbp-150h]
  int v95; // [rsp+30h] [rbp-150h]
  unsigned __int64 v96; // [rsp+30h] [rbp-150h]
  __int64 v98[2]; // [rsp+50h] [rbp-130h] BYREF
  unsigned __int64 v99; // [rsp+60h] [rbp-120h] BYREF
  unsigned int v100; // [rsp+68h] [rbp-118h]
  unsigned __int64 v101; // [rsp+70h] [rbp-110h] BYREF
  unsigned int v102; // [rsp+78h] [rbp-108h]
  unsigned __int64 v103; // [rsp+80h] [rbp-100h] BYREF
  unsigned int v104; // [rsp+88h] [rbp-F8h]
  unsigned __int64 v105; // [rsp+90h] [rbp-F0h] BYREF
  unsigned int v106; // [rsp+98h] [rbp-E8h]
  __int64 v107; // [rsp+A0h] [rbp-E0h] BYREF
  unsigned int v108; // [rsp+A8h] [rbp-D8h]
  unsigned __int64 v109; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned int v110; // [rsp+B8h] [rbp-C8h]
  __int64 v111[2]; // [rsp+C0h] [rbp-C0h] BYREF
  unsigned __int64 v112; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned int v113; // [rsp+D8h] [rbp-A8h]
  __int64 v114[2]; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned __int64 v115; // [rsp+F0h] [rbp-90h] BYREF
  unsigned int v116; // [rsp+F8h] [rbp-88h]
  __int64 v117; // [rsp+100h] [rbp-80h] BYREF
  unsigned int v118; // [rsp+108h] [rbp-78h]
  char v119; // [rsp+110h] [rbp-70h]
  unsigned __int64 v120; // [rsp+120h] [rbp-60h] BYREF
  unsigned int v121; // [rsp+128h] [rbp-58h]
  __int64 v122; // [rsp+130h] [rbp-50h] BYREF
  unsigned int v123; // [rsp+138h] [rbp-48h]
  char v124; // [rsp+140h] [rbp-40h]

  v3 = a1;
  v4 = a2;
  if ( *(_WORD *)(a2 + 24) )
  {
    v8 = sub_1456040(a2);
    v9 = sub_1456C90(a1, v8);
    sub_15897D0(&v105, v9, 1);
    v10 = sub_14687F0(a1, a2);
    if ( v10 )
    {
      if ( a3 )
      {
        sub_13D0020((__int64)&v103, v9);
        sub_13A38D0((__int64)&v109, (__int64)&v103);
        if ( v110 > 0x40 )
        {
          sub_16A5E70(&v109, v10);
        }
        else
        {
          v25 = (__int64)(v109 << (64 - (unsigned __int8)v110)) >> (64 - (unsigned __int8)v110);
          v26 = v25 >> v10;
          v27 = v25 >> 63;
          if ( v10 == v110 )
            v26 = v27;
          v109 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v110) & v26;
        }
        sub_13A38D0((__int64)&v112, (__int64)&v109);
        if ( v113 > 0x40 )
        {
          sub_16A7DC0(&v112, v10);
        }
        else
        {
          v28 = 0;
          if ( v10 != v113 )
            v28 = (v112 << v10) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v113);
          v112 = v28;
        }
        sub_16A7490(&v112, 1);
        v116 = v113;
        v113 = 0;
        v115 = v112;
        sub_13D00B0((__int64)&v101, v9);
      }
      else
      {
        sub_135E0D0((__int64)&v103, v9, -1, 1u);
        sub_13A38D0((__int64)&v109, (__int64)&v103);
        if ( v110 > 0x40 )
        {
          sub_16A8110(&v109, v10);
        }
        else if ( v10 == v110 )
        {
          v109 = 0;
        }
        else
        {
          v109 >>= v10;
        }
        sub_13A38D0((__int64)&v112, (__int64)&v109);
        if ( v113 > 0x40 )
        {
          sub_16A7DC0(&v112, v10);
        }
        else
        {
          v17 = 0;
          if ( v10 != v113 )
            v17 = (v112 << v10) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v113);
          v112 = v17;
        }
        sub_16A7490(&v112, 1);
        v18 = v113;
        v113 = 0;
        v116 = v18;
        v115 = v112;
        sub_135E0D0((__int64)&v101, v9, 0, 0);
      }
      sub_15898E0(&v120, &v101, &v115, v19, &v101);
      sub_1454200((__int64 *)&v105, (__int64 *)&v120);
      sub_135E100(&v122);
      sub_135E100((__int64 *)&v120);
      sub_135E100((__int64 *)&v101);
      sub_135E100((__int64 *)&v115);
      sub_135E100((__int64 *)&v112);
      sub_135E100((__int64 *)&v109);
      sub_135E100((__int64 *)&v103);
      v11 = *(_WORD *)(v4 + 24);
      if ( v11 == 4 )
        goto LABEL_6;
    }
    else
    {
      v11 = *(_WORD *)(a2 + 24);
      if ( v11 == 4 )
      {
LABEL_6:
        v12 = sub_1477920(a1, **(_QWORD **)(v4 + 32), a3);
        v116 = *(_DWORD *)(v12 + 8);
        if ( v116 > 0x40 )
          sub_16A4FD0(&v115, v12);
        else
          v115 = *(_QWORD *)v12;
        v118 = *(_DWORD *)(v12 + 24);
        if ( v118 > 0x40 )
          sub_16A4FD0(&v117, v12 + 16);
        else
          v117 = *(_QWORD *)(v12 + 16);
        v13 = 1;
        v90 = *(_QWORD *)(v4 + 40);
        if ( v90 != 1 )
        {
          do
          {
            v14 = sub_1477920(a1, *(_QWORD *)(*(_QWORD *)(v4 + 32) + 8LL * v13), a3);
            sub_158E130(&v120, &v115, v14);
            if ( v116 > 0x40 && v115 )
              j_j___libc_free_0_0(v115);
            v115 = v120;
            v15 = v121;
            v121 = 0;
            v116 = v15;
            if ( v118 > 0x40 && v117 )
              j_j___libc_free_0_0(v117);
            ++v13;
            v117 = v122;
            v16 = v123;
            v123 = 0;
            v118 = v16;
            sub_135E100(&v122);
            sub_135E100((__int64 *)&v120);
          }
          while ( v90 != v13 );
        }
        goto LABEL_18;
      }
    }
    switch ( v11 )
    {
      case 5:
        v20 = sub_1477920(a1, **(_QWORD **)(v4 + 32), a3);
        v116 = *(_DWORD *)(v20 + 8);
        if ( v116 > 0x40 )
          sub_16A4FD0(&v115, v20);
        else
          v115 = *(_QWORD *)v20;
        v118 = *(_DWORD *)(v20 + 24);
        if ( v118 > 0x40 )
          sub_16A4FD0(&v117, v20 + 16);
        else
          v117 = *(_QWORD *)(v20 + 16);
        v21 = 1;
        v91 = *(_QWORD *)(v4 + 40);
        if ( v91 != 1 )
        {
          do
          {
            v22 = sub_1477920(a1, *(_QWORD *)(*(_QWORD *)(v4 + 32) + 8LL * v21), a3);
            sub_158E820(&v120, &v115, v22);
            if ( v116 > 0x40 && v115 )
              j_j___libc_free_0_0(v115);
            v115 = v120;
            v23 = v121;
            v121 = 0;
            v116 = v23;
            if ( v118 > 0x40 && v117 )
              j_j___libc_free_0_0(v117);
            ++v21;
            v117 = v122;
            v24 = v123;
            v123 = 0;
            v118 = v24;
            sub_135E100(&v122);
            sub_135E100((__int64 *)&v120);
          }
          while ( v91 != v21 );
        }
        goto LABEL_18;
      case 9:
        v29 = 1;
        v30 = sub_1477920(a1, **(_QWORD **)(v4 + 32), a3);
        sub_1455FD0((__int64)&v115, v30);
        v92 = *(_QWORD *)(v4 + 40);
        if ( v92 == 1 )
        {
LABEL_18:
          sub_158BE00(&v120, &v105, &v115);
          v6 = sub_1463F10(v3, v4, a3, (__int64)&v120);
          sub_135E100(&v122);
          sub_135E100((__int64 *)&v120);
          sub_135E100(&v117);
          sub_135E100((__int64 *)&v115);
LABEL_19:
          sub_135E100(&v107);
          sub_135E100((__int64 *)&v105);
          return v6;
        }
        v31 = v4;
        v32 = a1;
        v33 = v31;
        do
        {
          v34 = v29++;
          v35 = sub_1477920(a1, *(_QWORD *)(*(_QWORD *)(v33 + 32) + 8 * v34), a3);
          sub_158F0F0(&v120, &v115, v35);
          sub_1454200((__int64 *)&v115, (__int64 *)&v120);
          sub_135E100(&v122);
          sub_135E100((__int64 *)&v120);
        }
        while ( v92 != v29 );
LABEL_57:
        v36 = v33;
        v3 = v32;
        v4 = v36;
        goto LABEL_18;
      case 8:
        v37 = 1;
        v38 = sub_1477920(a1, **(_QWORD **)(v4 + 32), a3);
        sub_1455FD0((__int64)&v115, v38);
        v93 = *(_QWORD *)(v4 + 40);
        if ( v93 == 1 )
          goto LABEL_18;
        v39 = v4;
        v32 = a1;
        v33 = v39;
        do
        {
          v40 = v37++;
          v41 = sub_1477920(a1, *(_QWORD *)(*(_QWORD *)(v33 + 32) + 8 * v40), a3);
          sub_158F360(&v120, &v115, v41);
          sub_1454200((__int64 *)&v115, (__int64 *)&v120);
          sub_135E100(&v122);
          sub_135E100((__int64 *)&v120);
        }
        while ( v93 != v37 );
        goto LABEL_57;
      case 6:
        v42 = sub_1477920(a1, *(_QWORD *)(v4 + 32), a3);
        sub_1455FD0((__int64)&v109, v42);
        v43 = sub_1477920(a1, *(_QWORD *)(v4 + 40), a3);
        sub_1455FD0((__int64)&v112, v43);
        sub_158FAB0(&v115, &v109, &v112);
        sub_158BE00(&v120, &v105, &v115);
        v6 = sub_1463F10(a1, v4, a3, (__int64)&v120);
        sub_135E100(&v122);
        sub_135E100((__int64 *)&v120);
        sub_135E100(&v117);
        sub_135E100((__int64 *)&v115);
        sub_135E100(v114);
        sub_135E100((__int64 *)&v112);
        sub_135E100(v111);
        sub_135E100((__int64 *)&v109);
        goto LABEL_19;
      case 2:
        v44 = sub_1477920(a1, *(_QWORD *)(v4 + 32), a3);
        sub_1455FD0((__int64)&v112, v44);
        sub_158CEA0(&v115, &v112, v9);
LABEL_77:
        sub_158BE00(&v120, &v105, &v115);
        v6 = sub_1463F10(a1, v4, a3, (__int64)&v120);
        sub_135E100(&v122);
        sub_135E100((__int64 *)&v120);
        sub_135E100(&v117);
        sub_135E100((__int64 *)&v115);
        sub_135E100(v114);
        sub_135E100((__int64 *)&v112);
        goto LABEL_19;
      case 3:
        v45 = sub_1477920(a1, *(_QWORD *)(v4 + 32), a3);
        sub_1455FD0((__int64)&v112, v45);
        sub_158D100(&v115, &v112, v9);
        goto LABEL_77;
      case 1:
        v46 = sub_1477920(a1, *(_QWORD *)(v4 + 32), a3);
        sub_1455FD0((__int64)&v112, v46);
        sub_158D430(&v115, &v112, v9);
        goto LABEL_77;
      case 7:
        v47 = *(_WORD *)(v4 + 26);
        if ( (v47 & 2) != 0 )
        {
          v48 = *(__int64 **)(v4 + 32);
          if ( !*(_WORD *)(*v48 + 24) )
          {
            v94 = *v48;
            if ( !sub_13D01C0(*(_QWORD *)(*v48 + 32) + 24LL) )
            {
              sub_135E0D0((__int64)&v112, v9, 0, 0);
              sub_13A38D0((__int64)&v109, *(_QWORD *)(v94 + 32) + 24LL);
              sub_15898E0(&v115, &v109, &v112, v82, &v109);
              sub_158BE00(&v120, &v105, &v115);
              sub_1454200((__int64 *)&v105, (__int64 *)&v120);
              sub_135E100(&v122);
              sub_135E100((__int64 *)&v120);
              sub_135E100(&v117);
              sub_135E100((__int64 *)&v115);
              sub_135E100((__int64 *)&v109);
              sub_135E100((__int64 *)&v112);
              v47 = *(_WORD *)(v4 + 26);
            }
          }
        }
        if ( (v47 & 4) == 0 )
          goto LABEL_97;
        v95 = *(_QWORD *)(v4 + 40);
        if ( !v95 )
          goto LABEL_148;
        v89 = v9;
        v49 = 1;
        v50 = 0;
        v51 = v4;
        v52 = 1;
        do
        {
          if ( !(unsigned __int8)sub_1477BC0(a1, *(_QWORD *)(*(_QWORD *)(v51 + 32) + 8 * v50)) )
            v49 = 0;
          if ( !(unsigned __int8)sub_1477A90(a1, *(_QWORD *)(*(_QWORD *)(v51 + 32) + 8 * v50)) )
            v52 = 0;
          ++v50;
        }
        while ( v95 != (_DWORD)v50 );
        v53 = v52;
        v9 = v89;
        v4 = v51;
        if ( v49 )
        {
LABEL_148:
          sub_13D00B0((__int64)&v112, v9);
          sub_135E0D0((__int64)&v109, v9, 0, 0);
        }
        else
        {
          if ( !v53 )
          {
LABEL_97:
            if ( *(_QWORD *)(v4 + 40) == 2 )
            {
              v96 = sub_1474260(a1, *(_QWORD *)(v4 + 48));
              if ( !sub_14562D0(v96) )
              {
                v83 = sub_1456040(v96);
                if ( sub_1456C90(a1, v83) <= (unsigned __int64)v9 )
                {
                  v84 = sub_13A5BC0((_QWORD *)v4, a1);
                  sub_1475920((__int64)&v112, a1, **(_QWORD **)(v4 + 32), v84, v96, v9);
                  if ( !(unsigned __int8)sub_158A0B0(&v112) )
                  {
                    sub_158BE00(&v120, &v105, &v112);
                    sub_1454200((__int64 *)&v105, (__int64 *)&v120);
                    sub_135E100(&v122);
                    sub_135E100((__int64 *)&v120);
                  }
                  v85 = sub_13A5BC0((_QWORD *)v4, a1);
                  sub_1475E30((__int64)&v115, a1, **(_QWORD **)(v4 + 32), v85, v96, v9);
                  if ( !(unsigned __int8)sub_158A0B0(&v115) )
                  {
                    sub_158BE00(&v120, &v105, &v115);
                    sub_1454200((__int64 *)&v105, (__int64 *)&v120);
                    sub_135E100(&v122);
                    sub_135E100((__int64 *)&v120);
                  }
                  sub_135E100(&v117);
                  sub_135E100((__int64 *)&v115);
                  sub_135E100(v114);
                  sub_135E100((__int64 *)&v112);
                }
              }
            }
            goto LABEL_98;
          }
          sub_135E0D0((__int64)&v112, v89, 1, 0);
          sub_13D00B0((__int64)&v109, v89);
        }
        sub_15898E0(&v115, &v109, &v112, v81, &v109);
        sub_158BE00(&v120, &v105, &v115);
        sub_1454200((__int64 *)&v105, (__int64 *)&v120);
        sub_135E100(&v122);
        sub_135E100((__int64 *)&v120);
        sub_135E100(&v117);
        sub_135E100((__int64 *)&v115);
        sub_135E100((__int64 *)&v109);
        sub_135E100((__int64 *)&v112);
        goto LABEL_97;
    }
    if ( v11 != 10 )
    {
LABEL_98:
      v54 = v106;
      v106 = 0;
      v121 = v54;
      v120 = v105;
      v55 = v108;
      v108 = 0;
      v123 = v55;
      v122 = v107;
      v6 = sub_1463F10(a1, v4, a3, (__int64)&v120);
      sub_135E100(&v122);
      sub_135E100((__int64 *)&v120);
      goto LABEL_19;
    }
    v56 = *(_QWORD *)(v4 - 8);
    if ( *(_BYTE *)(v56 + 16) > 0x17u
      && (*(_QWORD *)(v56 + 48) || *(__int16 *)(v56 + 18) < 0)
      && (v57 = sub_1625790(v56, 4)) != 0 )
    {
      sub_1593050(&v120, v57);
      v119 = 1;
      v116 = v121;
      v115 = v120;
      v118 = v123;
      v117 = v122;
      sub_158BE00(&v120, &v105, &v115);
      sub_1454200((__int64 *)&v105, (__int64 *)&v120);
      sub_135E100(&v122);
      sub_135E100((__int64 *)&v120);
    }
    else
    {
      v119 = 0;
    }
    v58 = sub_1632FA0(*(_QWORD *)(*(_QWORD *)(v3 + 24) + 40LL));
    if ( a3 )
    {
      v63 = sub_14C23D0(*(_QWORD *)(v4 - 8), v58, 0, *(_QWORD *)(v3 + 48), 0, *(_QWORD *)(v3 + 56));
      if ( v63 > 1 )
      {
        v64 = v63 - 1;
        sub_13D0020((__int64)&v101, v9);
        sub_13A38D0((__int64)&v103, (__int64)&v101);
        sub_14557D0((__int64)&v103, v64);
        sub_16A7490(&v103, 1);
        v110 = v104;
        v104 = 0;
        v109 = v103;
        sub_13D00B0((__int64)v98, v9);
        sub_13A38D0((__int64)&v99, (__int64)v98);
        sub_14557D0((__int64)&v99, v64);
        sub_15898E0(&v112, &v99, &v109, v65, v66);
        sub_158BE00(&v120, &v105, &v112);
        sub_1454200((__int64 *)&v105, (__int64 *)&v120);
        sub_135E100(&v122);
        sub_135E100((__int64 *)&v120);
        sub_135E100(v114);
        sub_135E100((__int64 *)&v112);
        sub_135E100((__int64 *)&v99);
        sub_135E100(v98);
        sub_135E100((__int64 *)&v109);
        sub_135E100((__int64 *)&v103);
        sub_135E100((__int64 *)&v101);
      }
    }
    else
    {
      sub_14C2530((unsigned int)&v109, *(_QWORD *)(v4 - 8), v58, 0, *(_QWORD *)(v3 + 48), 0, *(_QWORD *)(v3 + 56), 0);
      sub_13A38D0((__int64)&v103, (__int64)&v109);
      sub_13D0570((__int64)&v103);
      v59 = v104;
      v104 = 0;
      v113 = v59;
      v112 = v103;
      sub_16A7490(&v112, 1);
      v121 = v113;
      v113 = 0;
      v120 = v112;
      v87 = sub_1455820((__int64)v111, &v120);
      sub_135E100((__int64 *)&v120);
      sub_135E100((__int64 *)&v112);
      sub_135E100((__int64 *)&v103);
      if ( !v87 )
      {
        sub_13A38D0((__int64)&v99, (__int64)&v109);
        sub_13D0570((__int64)&v99);
        v102 = v100;
        v100 = 0;
        v101 = v99;
        sub_16A7490(&v101, 1);
        v104 = v102;
        v102 = 0;
        v103 = v101;
        sub_13A38D0((__int64)v98, (__int64)v111);
        sub_15898E0(&v112, v98, &v103, v73, v74);
        sub_158BE00(&v120, &v105, &v112);
        sub_1454200((__int64 *)&v105, (__int64 *)&v120);
        sub_135E100(&v122);
        sub_135E100((__int64 *)&v120);
        sub_135E100(v114);
        sub_135E100((__int64 *)&v112);
        sub_135E100(v98);
        sub_135E100((__int64 *)&v103);
        sub_135E100((__int64 *)&v101);
        sub_135E100((__int64 *)&v99);
      }
      sub_135E100(v111);
      sub_135E100((__int64 *)&v109);
    }
    v60 = *(_QWORD *)(v4 - 8);
    if ( *(_BYTE *)(v60 + 16) != 77 || (sub_145C630((__int64)&v120, v3 + 280, *(_QWORD *)(v4 - 8)), !v124) )
    {
LABEL_109:
      v61 = v106;
      v106 = 0;
      v121 = v61;
      v120 = v105;
      v62 = v108;
      v108 = 0;
      v123 = v62;
      v122 = v107;
      v6 = sub_1463F10(v3, v4, a3, (__int64)&v120);
      sub_135E100(&v122);
      sub_135E100((__int64 *)&v120);
      if ( v119 )
      {
        if ( v118 > 0x40 && v117 )
          j_j___libc_free_0_0(v117);
        if ( v116 > 0x40 && v115 )
          j_j___libc_free_0_0(v115);
      }
      goto LABEL_19;
    }
    sub_15897D0(&v109, v9, 0);
    if ( (*(_BYTE *)(v60 + 23) & 0x40) != 0 )
    {
      v67 = *(__int64 **)(v60 - 8);
      v68 = *(_DWORD *)(v60 + 20);
    }
    else
    {
      v68 = *(_DWORD *)(v60 + 20);
      v67 = (__int64 *)(v60 - 24LL * (v68 & 0xFFFFFFF));
    }
    v88 = v4;
    v69 = v67;
    v86 = &v67[3 * (v68 & 0xFFFFFFF)];
    while ( v69 != v86 )
    {
      v70 = *v69;
      v69 += 3;
      v71 = sub_146F1B0(v3, v70);
      v72 = sub_1477920(v3, v71, a3);
      sub_1455FD0((__int64)&v112, v72);
      sub_158C3A0(&v120, &v109, &v112);
      sub_1454200((__int64 *)&v109, (__int64 *)&v120);
      sub_135E100(&v122);
      sub_135E100((__int64 *)&v120);
      if ( (unsigned __int8)sub_158A0B0(&v109) )
      {
        v4 = v88;
        sub_135E100(v114);
        sub_135E100((__int64 *)&v112);
        goto LABEL_128;
      }
      sub_135E100(v114);
      sub_135E100((__int64 *)&v112);
    }
    v4 = v88;
LABEL_128:
    sub_158BE00(&v120, &v105, &v109);
    sub_1454200((__int64 *)&v105, (__int64 *)&v120);
    sub_135E100(&v122);
    sub_135E100((__int64 *)&v120);
    v75 = *(_QWORD **)(v3 + 296);
    v76 = *(_QWORD **)(v3 + 288);
    if ( v75 == v76 )
    {
      v80 = *(unsigned int *)(v3 + 308);
      while ( &v75[v80] != v76 && v60 != *v76 )
        ++v76;
    }
    else
    {
      v76 = (_QWORD *)sub_16CC9F0(v3 + 280, v60);
      if ( v60 == *v76 )
      {
        v77 = *(_QWORD **)(v3 + 296);
        v75 = *(_QWORD **)(v3 + 288);
        if ( v77 != v75 )
        {
          v78 = *(unsigned int *)(v3 + 304);
          goto LABEL_132;
        }
        v80 = *(unsigned int *)(v3 + 308);
      }
      else
      {
        v75 = *(_QWORD **)(v3 + 296);
        v77 = v75;
        if ( v75 != *(_QWORD **)(v3 + 288) )
        {
          v78 = *(unsigned int *)(v3 + 304);
          v76 = &v77[v78];
LABEL_132:
          v79 = &v77[v78];
LABEL_133:
          if ( v79 != v76 )
          {
            *v76 = -2;
            ++*(_DWORD *)(v3 + 312);
          }
          sub_135E100(v111);
          sub_135E100((__int64 *)&v109);
          goto LABEL_109;
        }
        v80 = *(unsigned int *)(v3 + 308);
        v76 = &v75[v80];
      }
    }
    v79 = &v75[v80];
    goto LABEL_133;
  }
  sub_13A38D0((__int64)&v115, *(_QWORD *)(a2 + 32) + 24LL);
  sub_1589870(&v120, &v115);
  v6 = sub_1463F10(a1, a2, a3, (__int64)&v120);
  sub_135E100(&v122);
  sub_135E100((__int64 *)&v120);
  sub_135E100((__int64 *)&v115);
  return v6;
}
