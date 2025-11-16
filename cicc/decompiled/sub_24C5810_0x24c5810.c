// Function: sub_24C5810
// Address: 0x24c5810
//
void __fastcall sub_24C5810(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rcx
  __int64 v11; // r12
  __int64 v12; // rax
  _BYTE *v13; // r13
  __int64 (__fastcall *v14)(__int64, __int64, __int64); // rax
  unsigned int *v15; // r12
  unsigned int *v16; // rbx
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  __int64 v21; // r15
  unsigned int i; // r12d
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // rax
  __int64 v26; // r8
  __int64 v27; // r9
  _BYTE *v28; // r13
  __int64 (__fastcall *v29)(__int64, __int64, __int64); // rax
  unsigned int *v30; // rbx
  unsigned int *v31; // r14
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  unsigned __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // r13
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdi
  __int64 **v46; // r13
  __int64 v47; // rax
  __int64 v48; // r8
  __int64 v49; // r9
  _BYTE *v50; // rbx
  __int64 (__fastcall *v51)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v52; // r12
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rbx
  __int64 v56; // rax
  unsigned __int64 v57; // rdx
  unsigned int v58; // eax
  __int64 (__fastcall *v59)(__int64, __int64, __int64); // rax
  unsigned int *v60; // r12
  unsigned int *v61; // rbx
  __int64 v62; // rdx
  unsigned int *v63; // r14
  unsigned int *v64; // rbx
  __int64 v65; // rdx
  __int64 (__fastcall *v66)(__int64, __int64, __int64); // rax
  __int64 v67; // rbx
  unsigned int *v68; // r12
  unsigned int *v69; // rbx
  __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // rdx
  int v73; // r13d
  unsigned int *v74; // rbx
  unsigned int *v75; // r13
  __int64 v76; // rdx
  unsigned int *v77; // r12
  unsigned int *v78; // rbx
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // r13
  __int64 *v82; // rdi
  __int64 *v83; // r15
  __int64 v84; // r12
  __int64 **v85; // rax
  __int64 v86; // rax
  unsigned int *v87; // r12
  unsigned int *v88; // rbx
  __int64 v89; // rdx
  unsigned int *v90; // r12
  unsigned int *j; // rbx
  __int64 v92; // rdx
  __int64 v93; // rdx
  __int64 v94; // rdx
  unsigned __int64 v95; // [rsp+8h] [rbp-278h]
  __int64 v97; // [rsp+28h] [rbp-258h]
  __int64 v98; // [rsp+30h] [rbp-250h]
  int v99; // [rsp+3Ch] [rbp-244h]
  __int64 *v100; // [rsp+40h] [rbp-240h]
  __int64 v101; // [rsp+48h] [rbp-238h]
  _BYTE v102[32]; // [rsp+50h] [rbp-230h] BYREF
  __int16 v103; // [rsp+70h] [rbp-210h]
  _BYTE v104[32]; // [rsp+80h] [rbp-200h] BYREF
  __int16 v105; // [rsp+A0h] [rbp-1E0h]
  unsigned int *v106; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned int v107; // [rsp+B8h] [rbp-1C8h]
  char v108; // [rsp+C0h] [rbp-1C0h] BYREF
  __int64 v109; // [rsp+E8h] [rbp-198h]
  __int64 v110; // [rsp+F0h] [rbp-190h]
  __int64 v111; // [rsp+100h] [rbp-180h]
  __int64 v112; // [rsp+108h] [rbp-178h]
  __int64 v113; // [rsp+110h] [rbp-170h]
  int v114; // [rsp+118h] [rbp-168h]
  void *v115; // [rsp+130h] [rbp-150h]
  __int64 *v116; // [rsp+140h] [rbp-140h] BYREF
  __int64 v117; // [rsp+148h] [rbp-138h]
  _BYTE v118[304]; // [rsp+150h] [rbp-130h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v116 = (__int64 *)v118;
  v117 = 0x2000000000LL;
  if ( v4 )
    v4 -= 24;
  v5 = sub_AA5190(v4);
  if ( v5 )
    v5 -= 24;
  sub_23D0AB0((__int64)&v106, v5, 0, 0, 0);
  v8 = *(_QWORD *)(a2 + 80);
  v95 = a2 + 72;
  if ( v8 != a2 + 72 )
  {
    v97 = *(_QWORD *)(a2 + 80);
    v9 = v97;
    if ( v8 )
      goto LABEL_7;
LABEL_60:
    v98 = 0;
    if ( !v8 )
    {
LABEL_61:
      v5 = a1[57];
      v103 = 257;
      v13 = (_BYTE *)a2;
      if ( v5 != *(_QWORD *)(a2 + 8) )
      {
        if ( *(_BYTE *)a2 > 0x15u )
        {
          v105 = 257;
          v13 = (_BYTE *)sub_B52210(a2, v5, (__int64)v104, 0, 0);
          v5 = (__int64)v13;
          (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
            v112,
            v13,
            v102,
            v109,
            v110);
          v87 = v106;
          v88 = &v106[4 * v107];
          if ( v106 != v88 )
          {
            do
            {
              v89 = *((_QWORD *)v87 + 1);
              v5 = *v87;
              v87 += 4;
              sub_B99FD0((__int64)v13, v5, v89);
            }
            while ( v88 != v87 );
          }
        }
        else
        {
          v59 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v111 + 136LL);
          if ( v59 == sub_928970 )
          {
            v13 = (_BYTE *)sub_ADAFB0(a2, v5);
          }
          else
          {
            v93 = v5;
            v5 = a2;
            v13 = (_BYTE *)v59(v111, a2, v93);
          }
          if ( *v13 > 0x1Cu )
          {
            v5 = (__int64)v13;
            (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
              v112,
              v13,
              v102,
              v109,
              v110);
            v60 = v106;
            v61 = &v106[4 * v107];
            if ( v106 != v61 )
            {
              do
              {
                v62 = *((_QWORD *)v60 + 1);
                v5 = *v60;
                v60 += 4;
                sub_B99FD0((__int64)v13, v5, v62);
              }
              while ( v61 != v60 );
            }
          }
        }
      }
      goto LABEL_16;
    }
    while ( 1 )
    {
      v11 = a1[57];
      v103 = 257;
      v12 = sub_ACC4F0(v98);
      v13 = (_BYTE *)v12;
      if ( v11 != *(_QWORD *)(v12 + 8) )
      {
        if ( *(_BYTE *)v12 > 0x15u )
        {
          v105 = 257;
          v13 = (_BYTE *)sub_B52210(v12, v11, (__int64)v104, 0, 0);
          v5 = (__int64)v13;
          (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
            v112,
            v13,
            v102,
            v109,
            v110);
          v77 = v106;
          v78 = &v106[4 * v107];
          if ( v106 != v78 )
          {
            do
            {
              v79 = *((_QWORD *)v77 + 1);
              v5 = *v77;
              v77 += 4;
              sub_B99FD0((__int64)v13, v5, v79);
            }
            while ( v78 != v77 );
          }
        }
        else
        {
          v14 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v111 + 136LL);
          if ( v14 == sub_928970 )
          {
            v5 = v11;
            v13 = (_BYTE *)sub_ADAFB0((unsigned __int64)v13, v11);
          }
          else
          {
            v5 = (__int64)v13;
            v13 = (_BYTE *)v14(v111, (__int64)v13, v11);
          }
          if ( *v13 > 0x1Cu )
          {
            v5 = (__int64)v13;
            (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
              v112,
              v13,
              v102,
              v109,
              v110);
            v15 = v106;
            v16 = &v106[4 * v107];
            if ( v106 != v16 )
            {
              do
              {
                v17 = *((_QWORD *)v15 + 1);
                v5 = *v15;
                v15 += 4;
                sub_B99FD0((__int64)v13, v5, v17);
              }
              while ( v16 != v15 );
            }
          }
        }
      }
LABEL_16:
      v18 = (unsigned int)v117;
      v19 = (unsigned int)v117 + 1LL;
      if ( v19 > HIDWORD(v117) )
      {
        v5 = (__int64)v118;
        sub_C8D5F0((__int64)&v116, v118, v19, 8u, v6, v7);
        v18 = (unsigned int)v117;
      }
      v116[v18] = (__int64)v13;
      LODWORD(v117) = v117 + 1;
      v101 = v98 + 48;
      v20 = *(_QWORD *)(v98 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v98 + 48 != v20 )
      {
        if ( !v20 )
          BUG();
        v21 = v20 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v20 - 24) - 30 <= 0xA )
        {
          v99 = sub_B46E30(v21);
          if ( v99 )
          {
            v100 = a1;
            for ( i = 0; i != v99; ++i )
            {
              v5 = i;
              v23 = sub_B46EC0(v21, i);
              v103 = 257;
              v24 = v100[57];
              v25 = sub_ACC4F0(v23);
              v28 = (_BYTE *)v25;
              if ( v24 != *(_QWORD *)(v25 + 8) )
              {
                if ( *(_BYTE *)v25 > 0x15u )
                {
                  v105 = 257;
                  v28 = (_BYTE *)sub_B52210(v25, v24, (__int64)v104, 0, 0);
                  v5 = (__int64)v28;
                  (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
                    v112,
                    v28,
                    v102,
                    v109,
                    v110);
                  v63 = v106;
                  v64 = &v106[4 * v107];
                  if ( v106 != v64 )
                  {
                    do
                    {
                      v65 = *((_QWORD *)v63 + 1);
                      v5 = *v63;
                      v63 += 4;
                      sub_B99FD0((__int64)v28, v5, v65);
                    }
                    while ( v64 != v63 );
                  }
                }
                else
                {
                  v29 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v111 + 136LL);
                  if ( v29 == sub_928970 )
                  {
                    v5 = v24;
                    v28 = (_BYTE *)sub_ADAFB0((unsigned __int64)v28, v24);
                  }
                  else
                  {
                    v5 = (__int64)v28;
                    v28 = (_BYTE *)v29(v111, (__int64)v28, v24);
                  }
                  if ( *v28 > 0x1Cu )
                  {
                    v5 = (__int64)v28;
                    (*(void (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
                      v112,
                      v28,
                      v102,
                      v109,
                      v110);
                    v30 = &v106[4 * v107];
                    if ( v106 != v30 )
                    {
                      v31 = v106;
                      do
                      {
                        v32 = *((_QWORD *)v31 + 1);
                        v5 = *v31;
                        v31 += 4;
                        sub_B99FD0((__int64)v28, v5, v32);
                      }
                      while ( v30 != v31 );
                    }
                  }
                }
              }
              v33 = (unsigned int)v117;
              v34 = (unsigned int)v117 + 1LL;
              if ( v34 > HIDWORD(v117) )
              {
                v5 = (__int64)v118;
                sub_C8D5F0((__int64)&v116, v118, v34, 8u, v26, v27);
                v33 = (unsigned int)v117;
              }
              v116[v33] = (__int64)v28;
              LODWORD(v117) = v117 + 1;
            }
            a1 = v100;
          }
        }
      }
      v35 = sub_AD6530(a1[57], v5);
      v38 = (unsigned int)v117;
      v39 = (unsigned int)v117 + 1LL;
      if ( v39 > HIDWORD(v117) )
      {
        v5 = (__int64)v118;
        sub_C8D5F0((__int64)&v116, v118, v39, 8u, v36, v37);
        v38 = (unsigned int)v117;
      }
      v116[v38] = v35;
      LODWORD(v117) = v117 + 1;
      v40 = *(_QWORD *)(v98 + 56);
      if ( v40 != v101 )
        break;
LABEL_56:
      v55 = sub_AD6530(a1[57], v5);
      v56 = (unsigned int)v117;
      v57 = (unsigned int)v117 + 1LL;
      if ( v57 > HIDWORD(v117) )
      {
        v5 = (__int64)v118;
        sub_C8D5F0((__int64)&v116, v118, v57, 8u, v6, v7);
        v56 = (unsigned int)v117;
      }
      v116[v56] = v55;
      v58 = v117 + 1;
      LODWORD(v117) = v117 + 1;
      v97 = *(_QWORD *)(v97 + 8);
      if ( v95 == v97 )
        goto LABEL_100;
      v9 = v97;
      v8 = *(_QWORD *)(a2 + 80);
      if ( !v97 )
        goto LABEL_60;
LABEL_7:
      v10 = v9 - 24;
      v98 = v10;
      if ( v8 && v10 == v8 - 24 )
        goto LABEL_61;
    }
    while ( 1 )
    {
      if ( !v40 )
        BUG();
      if ( (unsigned __int8)(*(_BYTE *)(v40 - 24) - 34) > 0x33u )
        goto LABEL_42;
      v42 = 0x8000000000041LL;
      if ( !_bittest64(&v42, (unsigned int)*(unsigned __int8 *)(v40 - 24) - 34) )
        goto LABEL_42;
      if ( sub_B491E0(v40 - 24) )
      {
        v45 = a1[58];
        v46 = (__int64 **)a1[57];
        v103 = 257;
        v5 = -1;
        v47 = sub_AD64C0(v45, -1, 0);
        v50 = (_BYTE *)v47;
        if ( v46 == *(__int64 ***)(v47 + 8) )
        {
          v52 = v47;
        }
        else
        {
          v51 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v111 + 120LL);
          if ( v51 == sub_920130 )
          {
            if ( *v50 > 0x15u )
              goto LABEL_85;
            v5 = (__int64)v50;
            if ( (unsigned __int8)sub_AC4810(0x30u) )
              v52 = sub_ADAB70(48, (unsigned __int64)v50, v46, 0);
            else
              v52 = sub_AA93C0(0x30u, (unsigned __int64)v50, (__int64)v46);
          }
          else
          {
            v5 = 48;
            v52 = v51(v111, 48u, v50, (__int64)v46);
          }
          if ( !v52 )
          {
LABEL_85:
            v105 = 257;
            v52 = sub_B51D30(48, (__int64)v50, (__int64)v46, (__int64)v104, 0, 0);
            if ( (unsigned __int8)sub_920620(v52) )
            {
              v73 = v114;
              if ( v113 )
                sub_B99FD0(v52, 3u, v113);
              sub_B45150(v52, v73);
            }
            v5 = v52;
            (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
              v112,
              v52,
              v102,
              v109,
              v110);
            v74 = v106;
            v75 = &v106[4 * v107];
            if ( v106 != v75 )
            {
              do
              {
                v76 = *((_QWORD *)v74 + 1);
                v5 = *v74;
                v74 += 4;
                sub_B99FD0(v52, v5, v76);
              }
              while ( v75 != v74 );
            }
          }
        }
        v53 = (unsigned int)v117;
        v54 = (unsigned int)v117 + 1LL;
        if ( v54 > HIDWORD(v117) )
        {
          v5 = (__int64)v118;
          sub_C8D5F0((__int64)&v116, v118, v54, 8u, v48, v49);
          v53 = (unsigned int)v117;
        }
        v116[v53] = v52;
        LODWORD(v117) = v117 + 1;
        v40 = *(_QWORD *)(v40 + 8);
        if ( v40 == v101 )
          goto LABEL_56;
      }
      else
      {
        v41 = *(_QWORD *)(v40 - 56);
        if ( v41
          && !*(_BYTE *)v41
          && *(_QWORD *)(v41 + 24) == *(_QWORD *)(v40 + 56)
          && (*(_BYTE *)(v41 + 33) & 0x20) == 0 )
        {
          v5 = a1[57];
          v103 = 257;
          if ( v5 != *(_QWORD *)(v41 + 8) )
          {
            if ( *(_BYTE *)v41 > 0x15u )
            {
              v105 = 257;
              v41 = sub_B52210(v41, v5, (__int64)v104, 0, 0);
              v5 = v41;
              (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
                v112,
                v41,
                v102,
                v109,
                v110);
              v90 = &v106[4 * v107];
              for ( j = v106; v90 != j; j += 4 )
              {
                v92 = *((_QWORD *)j + 1);
                v5 = *j;
                sub_B99FD0(v41, v5, v92);
              }
            }
            else
            {
              v66 = *(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v111 + 136LL);
              if ( v66 == sub_928970 )
              {
                v41 = sub_ADAFB0(v41, v5);
              }
              else
              {
                v94 = v5;
                v5 = v41;
                v41 = v66(v111, v41, v94);
              }
              if ( *(_BYTE *)v41 > 0x1Cu )
              {
                v5 = v41;
                (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v112 + 16LL))(
                  v112,
                  v41,
                  v102,
                  v109,
                  v110);
                v67 = 4LL * v107;
                v68 = &v106[v67];
                if ( v106 != &v106[v67] )
                {
                  v69 = v106;
                  do
                  {
                    v70 = *((_QWORD *)v69 + 1);
                    v5 = *v69;
                    v69 += 4;
                    sub_B99FD0(v41, v5, v70);
                  }
                  while ( v68 != v69 );
                }
              }
            }
          }
          v71 = (unsigned int)v117;
          v72 = (unsigned int)v117 + 1LL;
          if ( v72 > HIDWORD(v117) )
          {
            v5 = (__int64)v118;
            sub_C8D5F0((__int64)&v116, v118, v72, 8u, v43, v44);
            v71 = (unsigned int)v117;
          }
          v116[v71] = v41;
          LODWORD(v117) = v117 + 1;
        }
LABEL_42:
        v40 = *(_QWORD *)(v40 + 8);
        if ( v40 == v101 )
          goto LABEL_56;
      }
    }
  }
  v58 = v117;
LABEL_100:
  v80 = sub_24C54D0((__int64)a1, v58, a2, (__int64 *)a1[57], "sancov_cfs");
  v81 = (unsigned int)v117;
  v82 = (__int64 *)a1[57];
  a1[82] = v80;
  v83 = v116;
  v84 = v80;
  v85 = (__int64 **)sub_BCD420(v82, v81);
  v86 = sub_AD1300(v85, v83, v81);
  sub_B30160(v84, v86);
  *(_BYTE *)(a1[82] + 80) |= 1u;
  nullsub_61();
  v115 = &unk_49DA100;
  nullsub_63();
  if ( v106 != (unsigned int *)&v108 )
    _libc_free((unsigned __int64)v106);
  if ( v116 != (__int64 *)v118 )
    _libc_free((unsigned __int64)v116);
}
