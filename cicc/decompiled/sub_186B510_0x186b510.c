// Function: sub_186B510
// Address: 0x186b510
//
__int64 __fastcall sub_186B510(
        __int64 a1,
        __int64 a2,
        __int64 (__fastcall *a3)(__int64),
        __int64 a4,
        __int64 *a5,
        signed int a6,
        signed int a7,
        int a8,
        _DWORD *a9)
{
  __int64 v13; // rax
  unsigned __int64 v14; // r8
  int v15; // ecx
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rdi
  __int64 v22; // r10
  unsigned int i; // r14d
  __int64 v24; // rax
  int v25; // edx
  int *v27; // rax
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r9
  __int64 v32; // r8
  __int64 v33; // r15
  __int64 v34; // r15
  __int64 v35; // rbx
  _BYTE *v36; // r14
  _QWORD *v37; // rbx
  _QWORD *v38; // rdi
  _QWORD *v39; // r13
  _QWORD *v40; // rbx
  _QWORD *v41; // rdi
  char v42; // al
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // r9
  __int64 v46; // r8
  __int64 v47; // r15
  __int64 v48; // r15
  __int64 v49; // rbx
  _BYTE *v50; // r14
  _QWORD *v51; // rbx
  _QWORD *v52; // rdi
  _QWORD *v53; // rbx
  _QWORD *v54; // rdi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r9
  __int64 v58; // r8
  __int64 v59; // r14
  __int64 v60; // rbx
  _QWORD *v61; // rbx
  _QWORD *v62; // r14
  _QWORD *v63; // rdi
  _QWORD *v64; // rbx
  _QWORD *v65; // rdi
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // r9
  __int64 v69; // r8
  __int64 v70; // r14
  __int64 v71; // rbx
  _BYTE *v72; // r14
  _QWORD *v73; // rbx
  _QWORD *v74; // rdi
  _QWORD *v75; // rbx
  _QWORD *v76; // rdi
  __int64 v77; // rax
  __int64 v78; // rax
  char v79; // al
  __int64 v80; // rax
  __int64 v81; // rax
  char v82; // al
  __int64 v83; // rax
  __int64 v84; // rax
  char v85; // al
  __int64 v86; // rax
  __int64 v87; // rax
  char v88; // al
  int v89; // [rsp+8h] [rbp-528h]
  int v90; // [rsp+8h] [rbp-528h]
  unsigned __int64 v91; // [rsp+10h] [rbp-520h]
  __int64 v92; // [rsp+10h] [rbp-520h]
  unsigned __int64 v93; // [rsp+10h] [rbp-520h]
  __int64 v94; // [rsp+10h] [rbp-520h]
  unsigned __int64 v95; // [rsp+10h] [rbp-520h]
  __int64 v96; // [rsp+18h] [rbp-518h]
  __int64 v97; // [rsp+18h] [rbp-518h]
  __int64 v98; // [rsp+18h] [rbp-518h]
  __int64 v99; // [rsp+18h] [rbp-518h]
  __int64 v100; // [rsp+18h] [rbp-518h]
  _QWORD v101[2]; // [rsp+20h] [rbp-510h] BYREF
  _QWORD v102[2]; // [rsp+30h] [rbp-500h] BYREF
  _QWORD *v103; // [rsp+40h] [rbp-4F0h]
  _QWORD v104[6]; // [rsp+50h] [rbp-4E0h] BYREF
  _QWORD v105[2]; // [rsp+80h] [rbp-4B0h] BYREF
  _QWORD v106[2]; // [rsp+90h] [rbp-4A0h] BYREF
  _QWORD *v107; // [rsp+A0h] [rbp-490h]
  _QWORD v108[6]; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD v109[2]; // [rsp+E0h] [rbp-450h] BYREF
  _QWORD v110[2]; // [rsp+F0h] [rbp-440h] BYREF
  _QWORD *v111; // [rsp+100h] [rbp-430h]
  _QWORD v112[6]; // [rsp+110h] [rbp-420h] BYREF
  void *v113; // [rsp+140h] [rbp-3F0h] BYREF
  int v114; // [rsp+148h] [rbp-3E8h]
  char v115; // [rsp+14Ch] [rbp-3E4h]
  __int64 v116; // [rsp+150h] [rbp-3E0h]
  __m128i v117; // [rsp+158h] [rbp-3D8h]
  __int64 v118; // [rsp+168h] [rbp-3C8h]
  __int64 v119; // [rsp+170h] [rbp-3C0h]
  __m128i v120; // [rsp+178h] [rbp-3B8h]
  __int64 v121; // [rsp+188h] [rbp-3A8h]
  char v122; // [rsp+190h] [rbp-3A0h]
  _BYTE *v123; // [rsp+198h] [rbp-398h] BYREF
  __int64 v124; // [rsp+1A0h] [rbp-390h]
  _BYTE v125[352]; // [rsp+1A8h] [rbp-388h] BYREF
  char v126; // [rsp+308h] [rbp-228h]
  int v127; // [rsp+30Ch] [rbp-224h]
  __int64 v128; // [rsp+310h] [rbp-220h]
  _QWORD v129[11]; // [rsp+320h] [rbp-210h] BYREF
  _QWORD *v130; // [rsp+378h] [rbp-1B8h]
  unsigned int v131; // [rsp+380h] [rbp-1B0h]
  _BYTE v132[424]; // [rsp+388h] [rbp-1A8h] BYREF

  v13 = a3(a4);
  v14 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v15 = v13;
  v16 = v13 >> 32;
  v17 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 72;
  if ( (a2 & 4) != 0 )
    v17 = (a2 & 0xFFFFFFFFFFFFFFF8LL) - 24;
  v18 = *(_QWORD *)v17;
  if ( *(_BYTE *)(*(_QWORD *)v17 + 16LL) )
    v18 = 0;
  if ( v15 == 0x80000000 )
  {
    *(_BYTE *)(a1 + 8) = 1;
    *(_DWORD *)a1 = 0x80000000;
    *(_DWORD *)(a1 + 4) = v16;
  }
  else
  {
    v19 = *(_QWORD *)(v14 + 40);
    v20 = *(_QWORD *)(v19 + 56);
    if ( v15 == 0x7FFFFFFF )
    {
      v95 = v14;
      v98 = *(_QWORD *)(v19 + 56);
      if ( (unsigned __int8)sub_186A5D0(v18, v98) )
        goto LABEL_100;
      v55 = sub_15E0530(*a5);
      v56 = sub_1602790(v55);
      v57 = v98;
      v58 = v95;
      if ( !v56 )
      {
        v77 = sub_15E0530(*a5);
        v78 = sub_16033E0(v77);
        v79 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v78 + 48LL))(v78);
        v57 = v98;
        v58 = v95;
        if ( !v79 )
          goto LABEL_100;
      }
      v99 = v57;
      sub_15CA5C0((__int64)v129, (__int64)"inline", (__int64)"NeverInline", 11, v58);
      sub_15C9340((__int64)v109, "Callee", 6u, v18);
      v59 = sub_186AF50((__int64)v129, (__int64)v109);
      sub_15CAB20(v59, " not inlined into ", 0x12u);
      sub_15C9340((__int64)v105, "Caller", 6u, v99);
      v60 = sub_17C21B0(v59, (__int64)v105);
      sub_15CAB20(v60, " because it should never be inlined (cost=never)", 0x30u);
      v114 = *(_DWORD *)(v60 + 8);
      v115 = *(_BYTE *)(v60 + 12);
      v116 = *(_QWORD *)(v60 + 16);
      v117 = _mm_loadu_si128((const __m128i *)(v60 + 24));
      v118 = *(_QWORD *)(v60 + 40);
      v113 = &unk_49ECF68;
      v119 = *(_QWORD *)(v60 + 48);
      v120 = _mm_loadu_si128((const __m128i *)(v60 + 56));
      v122 = *(_BYTE *)(v60 + 80);
      if ( v122 )
        v121 = *(_QWORD *)(v60 + 72);
      v123 = v125;
      v124 = 0x400000000LL;
      if ( *(_DWORD *)(v60 + 96) )
        sub_186B280((__int64)&v123, v60 + 88);
      v126 = *(_BYTE *)(v60 + 456);
      v127 = *(_DWORD *)(v60 + 460);
      v128 = *(_QWORD *)(v60 + 464);
      v113 = &unk_49ECFC8;
      if ( v107 != v108 )
        j_j___libc_free_0(v107, v108[0] + 1LL);
      if ( (_QWORD *)v105[0] != v106 )
        j_j___libc_free_0(v105[0], v106[0] + 1LL);
      if ( v111 != v112 )
        j_j___libc_free_0(v111, v112[0] + 1LL);
      if ( (_QWORD *)v109[0] != v110 )
        j_j___libc_free_0(v109[0], v110[0] + 1LL);
      v61 = v130;
      v129[0] = &unk_49ECF68;
      v62 = &v130[11 * v131];
      if ( v130 != v62 )
      {
        do
        {
          v62 -= 11;
          v63 = (_QWORD *)v62[4];
          if ( v63 != v62 + 6 )
            j_j___libc_free_0(v63, v62[6] + 1LL);
          if ( (_QWORD *)*v62 != v62 + 2 )
            j_j___libc_free_0(*v62, v62[2] + 1LL);
        }
        while ( v61 != v62 );
        v62 = v130;
      }
      if ( v62 != (_QWORD *)v132 )
        _libc_free((unsigned __int64)v62);
      sub_143AA50(a5, (__int64)&v113);
      v64 = v123;
      v113 = &unk_49ECF68;
      v39 = &v123[88 * (unsigned int)v124];
      if ( v123 != (_BYTE *)v39 )
      {
        do
        {
          v39 -= 11;
          v65 = (_QWORD *)v39[4];
          if ( v65 != v39 + 6 )
            j_j___libc_free_0(v65, v39[6] + 1LL);
          if ( (_QWORD *)*v39 != v39 + 2 )
            j_j___libc_free_0(*v39, v39[2] + 1LL);
        }
        while ( v64 != v39 );
        goto LABEL_59;
      }
      goto LABEL_60;
    }
    if ( byte_4FABC40 )
    {
      *(_BYTE *)(a1 + 8) = 1;
      *(_QWORD *)a1 = 0x80000000LL;
    }
    else
    {
      if ( v18 )
      {
        v21 = *(_QWORD *)(v18 + 80);
        v22 = v18 + 72;
        for ( i = 0; v22 != v21; i += v25 )
        {
          while ( 1 )
          {
            if ( !v21 )
              BUG();
            v24 = *(_QWORD *)(v21 + 24);
            if ( v21 + 16 != v24 )
              break;
            v21 = *(_QWORD *)(v21 + 8);
            if ( v22 == v21 )
              goto LABEL_15;
          }
          v25 = 0;
          do
          {
            v24 = *(_QWORD *)(v24 + 8);
            ++v25;
          }
          while ( v21 + 16 != v24 );
          v21 = *(_QWORD *)(v21 + 8);
        }
      }
      else
      {
        i = -1;
      }
LABEL_15:
      if ( v15 >= (int)v16 )
      {
        v89 = v15;
        v91 = v14;
        v96 = v20;
        v27 = (int *)sub_16D40F0((__int64)qword_4FBB410);
        v20 = v96;
        v14 = v91;
        v15 = v89;
        if ( v27 )
          v28 = *v27;
        else
          v28 = qword_4FBB410[2];
        if ( v28 <= 2 )
        {
          if ( (unsigned __int8)sub_186A5D0(v18, v96) )
            goto LABEL_100;
          v66 = sub_15E0530(*a5);
          v67 = sub_1602790(v66);
          v68 = v96;
          v69 = v91;
          if ( !v67 )
          {
            v80 = sub_15E0530(*a5);
            v81 = sub_16033E0(v80);
            v82 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v81 + 48LL))(v81);
            v68 = v96;
            v69 = v91;
            if ( !v82 )
              goto LABEL_100;
          }
          v100 = v68;
          sub_15CA5C0((__int64)v129, (__int64)"inline", (__int64)"OptLevel", 8, v69);
          sub_15C9340((__int64)v109, "Callee", 6u, v18);
          v70 = sub_186AF50((__int64)v129, (__int64)v109);
          sub_15CAB20(v70, " not inlined into ", 0x12u);
          sub_15C9340((__int64)v105, "Caller", 6u, v100);
          v71 = sub_17C21B0(v70, (__int64)v105);
          sub_15CAB20(v71, " because opt level doesn't allow aggressive inlining", 0x34u);
          v114 = *(_DWORD *)(v71 + 8);
          v115 = *(_BYTE *)(v71 + 12);
          v116 = *(_QWORD *)(v71 + 16);
          v117 = _mm_loadu_si128((const __m128i *)(v71 + 24));
          v118 = *(_QWORD *)(v71 + 40);
          v113 = &unk_49ECF68;
          v119 = *(_QWORD *)(v71 + 48);
          v120 = _mm_loadu_si128((const __m128i *)(v71 + 56));
          v122 = *(_BYTE *)(v71 + 80);
          if ( v122 )
            v121 = *(_QWORD *)(v71 + 72);
          v123 = v125;
          v124 = 0x400000000LL;
          if ( *(_DWORD *)(v71 + 96) )
            sub_186B280((__int64)&v123, v71 + 88);
          v126 = *(_BYTE *)(v71 + 456);
          v127 = *(_DWORD *)(v71 + 460);
          v128 = *(_QWORD *)(v71 + 464);
          v113 = &unk_49ECFC8;
          if ( v107 != v108 )
            j_j___libc_free_0(v107, v108[0] + 1LL);
          if ( (_QWORD *)v105[0] != v106 )
            j_j___libc_free_0(v105[0], v106[0] + 1LL);
          if ( v111 != v112 )
            j_j___libc_free_0(v111, v112[0] + 1LL);
          if ( (_QWORD *)v109[0] != v110 )
            j_j___libc_free_0(v109[0], v110[0] + 1LL);
          v72 = v130;
          v129[0] = &unk_49ECF68;
          v73 = &v130[11 * v131];
          if ( v130 != v73 )
          {
            do
            {
              v73 -= 11;
              v74 = (_QWORD *)v73[4];
              if ( v74 != v73 + 6 )
                j_j___libc_free_0(v74, v73[6] + 1LL);
              if ( (_QWORD *)*v73 != v73 + 2 )
                j_j___libc_free_0(*v73, v73[2] + 1LL);
            }
            while ( v72 != (_BYTE *)v73 );
            v72 = v130;
          }
          if ( v72 != v132 )
            _libc_free((unsigned __int64)v72);
          sub_143AA50(a5, (__int64)&v113);
          v39 = v123;
          v113 = &unk_49ECF68;
          v75 = &v123[88 * (unsigned int)v124];
          if ( v123 != (_BYTE *)v75 )
          {
            do
            {
              v75 -= 11;
              v76 = (_QWORD *)v75[4];
              if ( v76 != v75 + 6 )
                j_j___libc_free_0(v76, v75[6] + 1LL);
              if ( (_QWORD *)*v75 != v75 + 2 )
                j_j___libc_free_0(*v75, v75[2] + 1LL);
            }
            while ( v39 != v75 );
            goto LABEL_59;
          }
          goto LABEL_60;
        }
        if ( a7 / 100 < (int)i && (int)(i + *a9) > a7 )
        {
          if ( (unsigned __int8)sub_186A5D0(v18, v96) )
            goto LABEL_100;
          v29 = sub_15E0530(*a5);
          v30 = sub_1602790(v29);
          v31 = v96;
          v32 = v91;
          if ( !v30 )
          {
            v86 = sub_15E0530(*a5);
            v87 = sub_16033E0(v86);
            v88 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v87 + 48LL))(v87);
            v31 = v96;
            v32 = v91;
            if ( !v88 )
              goto LABEL_100;
          }
          v92 = v31;
          sub_15CA5C0((__int64)v129, (__int64)"inline", (__int64)"TooBig", 6, v32);
          sub_15C9340((__int64)v109, "Callee", 6u, v18);
          v33 = sub_186AF50((__int64)v129, (__int64)v109);
          sub_15CAB20(v33, " not inlined into ", 0x12u);
          sub_15C9340((__int64)v105, "Caller", 6u, v92);
          v34 = sub_17C21B0(v33, (__int64)v105);
          sub_15CAB20(v34, " because callee doesn't have forceinline", 0x28u);
          sub_15CAB20(v34, " attribute and is too big for auto inlining (CalleeSize=", 0x38u);
          sub_15C9890((__int64)v101, "CalleeSize", 10, i);
          v35 = sub_17C21B0(v34, (__int64)v101);
          sub_15CAB20(v35, ")", 1u);
          v114 = *(_DWORD *)(v35 + 8);
          v115 = *(_BYTE *)(v35 + 12);
          v116 = *(_QWORD *)(v35 + 16);
          v117 = _mm_loadu_si128((const __m128i *)(v35 + 24));
          v118 = *(_QWORD *)(v35 + 40);
          v113 = &unk_49ECF68;
          v119 = *(_QWORD *)(v35 + 48);
          v120 = _mm_loadu_si128((const __m128i *)(v35 + 56));
          v122 = *(_BYTE *)(v35 + 80);
          if ( v122 )
            v121 = *(_QWORD *)(v35 + 72);
          v123 = v125;
          v124 = 0x400000000LL;
          if ( *(_DWORD *)(v35 + 96) )
            sub_186B280((__int64)&v123, v35 + 88);
          v126 = *(_BYTE *)(v35 + 456);
          v127 = *(_DWORD *)(v35 + 460);
          v128 = *(_QWORD *)(v35 + 464);
          v113 = &unk_49ECFC8;
          if ( v103 != v104 )
            j_j___libc_free_0(v103, v104[0] + 1LL);
          if ( (_QWORD *)v101[0] != v102 )
            j_j___libc_free_0(v101[0], v102[0] + 1LL);
          if ( v107 != v108 )
            j_j___libc_free_0(v107, v108[0] + 1LL);
          if ( (_QWORD *)v105[0] != v106 )
            j_j___libc_free_0(v105[0], v106[0] + 1LL);
          if ( v111 != v112 )
            j_j___libc_free_0(v111, v112[0] + 1LL);
          if ( (_QWORD *)v109[0] != v110 )
            j_j___libc_free_0(v109[0], v110[0] + 1LL);
          v36 = v130;
          v129[0] = &unk_49ECF68;
          v37 = &v130[11 * v131];
          if ( v130 != v37 )
          {
            do
            {
              v37 -= 11;
              v38 = (_QWORD *)v37[4];
              if ( v38 != v37 + 6 )
                j_j___libc_free_0(v38, v37[6] + 1LL);
              if ( (_QWORD *)*v37 != v37 + 2 )
                j_j___libc_free_0(*v37, v37[2] + 1LL);
            }
            while ( v36 != (_BYTE *)v37 );
            v36 = v130;
          }
          if ( v36 != v132 )
            _libc_free((unsigned __int64)v36);
          sub_143AA50(a5, (__int64)&v113);
          v39 = v123;
          v113 = &unk_49ECF68;
          v40 = &v123[88 * (unsigned int)v124];
          if ( v123 != (_BYTE *)v40 )
          {
            do
            {
              v40 -= 11;
              v41 = (_QWORD *)v40[4];
              if ( v41 != v40 + 6 )
                j_j___libc_free_0(v41, v40[6] + 1LL);
              if ( (_QWORD *)*v40 != v40 + 2 )
                j_j___libc_free_0(*v40, v40[2] + 1LL);
            }
            while ( v39 != v40 );
            goto LABEL_59;
          }
LABEL_60:
          if ( v39 != (_QWORD *)v125 )
            _libc_free((unsigned __int64)v39);
          goto LABEL_100;
        }
      }
      if ( (int)(i + a8) > a6 )
      {
        v90 = v15;
        v97 = v20;
        v93 = v14;
        v42 = sub_1C2F070(v20);
        v15 = v90;
        if ( !v42 )
        {
          if ( (unsigned __int8)sub_186A5D0(v18, v97) )
            goto LABEL_100;
          v43 = sub_15E0530(*a5);
          v44 = sub_1602790(v43);
          v45 = v97;
          v46 = v93;
          if ( !v44 )
          {
            v83 = sub_15E0530(*a5);
            v84 = sub_16033E0(v83);
            v85 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v84 + 48LL))(v84);
            v45 = v97;
            v46 = v93;
            if ( !v85 )
            {
LABEL_100:
              *(_BYTE *)(a1 + 8) = 0;
              return a1;
            }
          }
          v94 = v45;
          sub_15CA5C0((__int64)v129, (__int64)"inline", (__int64)"TooCostly", 9, v46);
          sub_15C9340((__int64)v109, "Callee", 6u, v18);
          v47 = sub_186AF50((__int64)v129, (__int64)v109);
          sub_15CAB20(v47, " not inlined into ", 0x12u);
          sub_15C9340((__int64)v105, "Caller", 6u, v94);
          v48 = sub_17C21B0(v47, (__int64)v105);
          sub_15CAB20(v48, " because callee doesn't have forceinline", 0x28u);
          sub_15CAB20(v48, " attribute and inlining it would exceed total Inline Budget.", 0x3Cu);
          sub_15CAB20(v48, " (CalleeSize = ", 0xFu);
          sub_15C9890((__int64)v101, "CalleeSize", 10, i);
          v49 = sub_17C21B0(v48, (__int64)v101);
          sub_15CAB20(v49, ")", 1u);
          v114 = *(_DWORD *)(v49 + 8);
          v115 = *(_BYTE *)(v49 + 12);
          v116 = *(_QWORD *)(v49 + 16);
          v117 = _mm_loadu_si128((const __m128i *)(v49 + 24));
          v118 = *(_QWORD *)(v49 + 40);
          v113 = &unk_49ECF68;
          v119 = *(_QWORD *)(v49 + 48);
          v120 = _mm_loadu_si128((const __m128i *)(v49 + 56));
          v122 = *(_BYTE *)(v49 + 80);
          if ( v122 )
            v121 = *(_QWORD *)(v49 + 72);
          v123 = v125;
          v124 = 0x400000000LL;
          if ( *(_DWORD *)(v49 + 96) )
            sub_186B280((__int64)&v123, v49 + 88);
          v126 = *(_BYTE *)(v49 + 456);
          v127 = *(_DWORD *)(v49 + 460);
          v128 = *(_QWORD *)(v49 + 464);
          v113 = &unk_49ECFC8;
          if ( v103 != v104 )
            j_j___libc_free_0(v103, v104[0] + 1LL);
          if ( (_QWORD *)v101[0] != v102 )
            j_j___libc_free_0(v101[0], v102[0] + 1LL);
          if ( v107 != v108 )
            j_j___libc_free_0(v107, v108[0] + 1LL);
          if ( (_QWORD *)v105[0] != v106 )
            j_j___libc_free_0(v105[0], v106[0] + 1LL);
          if ( v111 != v112 )
            j_j___libc_free_0(v111, v112[0] + 1LL);
          if ( (_QWORD *)v109[0] != v110 )
            j_j___libc_free_0(v109[0], v110[0] + 1LL);
          v50 = v130;
          v129[0] = &unk_49ECF68;
          v51 = &v130[11 * v131];
          if ( v130 != v51 )
          {
            do
            {
              v51 -= 11;
              v52 = (_QWORD *)v51[4];
              if ( v52 != v51 + 6 )
                j_j___libc_free_0(v52, v51[6] + 1LL);
              if ( (_QWORD *)*v51 != v51 + 2 )
                j_j___libc_free_0(*v51, v51[2] + 1LL);
            }
            while ( v50 != (_BYTE *)v51 );
            v50 = v130;
          }
          if ( v50 != v132 )
            _libc_free((unsigned __int64)v50);
          sub_143AA50(a5, (__int64)&v113);
          v39 = v123;
          v113 = &unk_49ECF68;
          v53 = &v123[88 * (unsigned int)v124];
          if ( v123 != (_BYTE *)v53 )
          {
            do
            {
              v53 -= 11;
              v54 = (_QWORD *)v53[4];
              if ( v54 != v53 + 6 )
                j_j___libc_free_0(v54, v53[6] + 1LL);
              if ( (_QWORD *)*v53 != v53 + 2 )
                j_j___libc_free_0(*v53, v53[2] + 1LL);
            }
            while ( v39 != v53 );
LABEL_59:
            v39 = v123;
            goto LABEL_60;
          }
          goto LABEL_60;
        }
      }
      *a9 += i;
      *(_BYTE *)(a1 + 8) = 1;
      *(_DWORD *)a1 = v15;
      *(_DWORD *)(a1 + 4) = v16;
    }
  }
  return a1;
}
