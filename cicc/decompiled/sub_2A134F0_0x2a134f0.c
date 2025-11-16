// Function: sub_2A134F0
// Address: 0x2a134f0
//
__int64 __fastcall sub_2A134F0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 (__fastcall *a7)(__int64),
        __int64 a8)
{
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // r9
  _QWORD *v17; // rbx
  _QWORD *v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  _BYTE *v21; // r12
  _QWORD *v22; // rdx
  int v23; // eax
  __int64 v24; // r15
  __int64 *v25; // rdx
  __int64 v26; // r13
  __int64 v27; // rsi
  _QWORD *v28; // rax
  _QWORD *v29; // rdx
  __int64 v30; // r12
  int v31; // r14d
  __int64 v32; // r15
  _QWORD *v33; // rax
  __int64 v34; // r9
  _QWORD *v35; // rax
  __int64 v36; // rcx
  _QWORD *v37; // rdx
  _BYTE *v38; // r13
  _QWORD *v39; // rcx
  _QWORD *v40; // rax
  _BYTE *v41; // rbx
  __int64 *v42; // r14
  int v43; // ebx
  __int64 v44; // rax
  __int64 *v45; // r12
  int v47; // ebx
  int v48; // ebx
  __int64 v49; // rax
  __int64 v50; // r13
  __int64 v51; // r15
  __int64 v52; // r12
  __int64 v53; // r9
  __int64 *v54; // r14
  unsigned int v55; // ecx
  __int64 **v56; // rdx
  __int64 *v57; // rdi
  __int64 *v58; // rax
  __int64 v59; // rdi
  int v60; // esi
  __int64 v61; // r8
  __int64 v62; // rsi
  __int64 v63; // rdi
  int v64; // ecx
  __int64 v65; // r9
  int v66; // ecx
  unsigned int v67; // edx
  __int64 *v68; // rax
  __int64 v69; // r10
  _QWORD *v70; // rax
  unsigned int v71; // esi
  __int64 *v72; // rdx
  __int64 v73; // r10
  _QWORD *v74; // rdx
  __int64 **v75; // rdx
  int v76; // r15d
  unsigned int v77; // edi
  __int64 **v78; // rax
  __int64 *v79; // r9
  __int64 *v80; // r15
  __int64 v81; // rax
  int v82; // edx
  int v83; // edx
  int v84; // r11d
  int v85; // eax
  int v86; // r11d
  __int64 v87; // rcx
  _QWORD *v88; // rdx
  _QWORD *v89; // rcx
  _QWORD *v90; // rax
  _BYTE *v91; // rbx
  __int64 *v92; // r14
  int v93; // edi
  unsigned int v94; // r9d
  __int64 *v95; // rcx
  __int64 **v96; // rax
  int v97; // r10d
  int v98; // r10d
  __int64 v99; // r9
  __int64 *v100; // rcx
  __int64 v101; // [rsp+0h] [rbp-100h]
  __int64 v102; // [rsp+8h] [rbp-F8h]
  __int64 v103; // [rsp+10h] [rbp-F0h]
  _QWORD *v105; // [rsp+28h] [rbp-D8h]
  __int64 v109; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v110; // [rsp+58h] [rbp-A8h] BYREF
  __int64 v111; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v112; // [rsp+68h] [rbp-98h]
  __int64 v113; // [rsp+70h] [rbp-90h]
  unsigned int v114; // [rsp+78h] [rbp-88h]
  _QWORD *v115; // [rsp+80h] [rbp-80h]
  _BYTE *v116; // [rsp+90h] [rbp-70h] BYREF
  __int64 v117; // [rsp+98h] [rbp-68h]
  _BYTE v118[96]; // [rsp+A0h] [rbp-60h] BYREF

  v116 = v118;
  v117 = 0x600000000LL;
  v8 = *(__int64 **)(a1 + 32);
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v9 = *v8;
  if ( v9 )
  {
    v10 = (unsigned int)(*(_DWORD *)(v9 + 44) + 1);
    v11 = *(_DWORD *)(v9 + 44) + 1;
  }
  else
  {
    v10 = 0;
    v11 = 0;
  }
  if ( *(_DWORD *)(a2 + 32) <= v11 )
    BUG();
  v12 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8 * v10);
  v13 = *(_QWORD *)(v12 + 24);
  v14 = v13 + 8LL * *(unsigned int *)(v12 + 32);
  v15 = (_QWORD *)sub_22077B0(0x40u);
  v17 = v15;
  if ( v15 )
  {
    v15[2] = 0;
    *v15 = &v111;
    v18 = v115;
    v17[3] = 0;
    v17[1] = v18;
    v115 = v17;
    v17[4] = v12;
    v17[5] = v13;
    v17[6] = v14;
    *((_BYTE *)v17 + 56) = 0;
  }
  v19 = (unsigned int)v117;
  v20 = v117;
  if ( HIDWORD(v117) > (unsigned int)v117 )
    goto LABEL_7;
  v21 = (_BYTE *)sub_C8D7D0((__int64)&v116, (__int64)v118, 0, 8u, &v110, v16);
  v87 = 8LL * (unsigned int)v117;
  if ( &v21[v87] )
  {
    *(_QWORD *)&v21[v87] = v17;
    v87 = 8LL * (unsigned int)v117;
  }
  v88 = v116;
  v38 = &v116[v87];
  if ( v116 != &v116[v87] )
  {
    v89 = &v21[v87];
    v90 = v21;
    do
    {
      if ( v90 )
      {
        *v90 = *v88;
        *v88 = 0;
      }
      ++v90;
      ++v88;
    }
    while ( v90 != v89 );
    v91 = v116;
    v38 = &v116[8 * (unsigned int)v117];
    if ( v116 != v38 )
    {
      do
      {
        v92 = (__int64 *)*((_QWORD *)v38 - 1);
        v38 -= 8;
        if ( v92 )
        {
          sub_2A131D0(v92);
          j_j___libc_free_0((unsigned __int64)v92);
        }
      }
      while ( v91 != v38 );
LABEL_32:
      v38 = v116;
    }
  }
LABEL_33:
  v43 = v110;
  if ( v38 != v118 )
    _libc_free((unsigned __int64)v38);
  v116 = v21;
  v20 = v117;
  HIDWORD(v117) = v43;
LABEL_9:
  v23 = v20 + 1;
  LODWORD(v117) = v23;
LABEL_10:
  if ( v23 )
  {
    while ( 1 )
    {
      v24 = *(_QWORD *)&v21[8 * v23 - 8];
      if ( !*(_BYTE *)(v24 + 56) )
        break;
      v25 = *(__int64 **)(v24 + 40);
      if ( *(__int64 **)(v24 + 48) == v25 )
      {
        v44 = (unsigned int)(v23 - 1);
        LODWORD(v117) = v44;
        v45 = *(__int64 **)&v21[8 * v44];
        if ( v45 )
        {
          sub_2A131D0(v45);
          j_j___libc_free_0((unsigned __int64)v45);
        }
      }
      else
      {
        v26 = *v25;
        *(_QWORD *)(v24 + 40) = v25 + 1;
        v27 = *(_QWORD *)v26;
        if ( *(_BYTE *)(a1 + 84) )
        {
          v28 = *(_QWORD **)(a1 + 64);
          v29 = &v28[*(unsigned int *)(a1 + 76)];
          if ( v28 == v29 )
          {
LABEL_42:
            v23 = v117;
            goto LABEL_10;
          }
          while ( v27 != *v28 )
          {
            if ( v29 == ++v28 )
              goto LABEL_42;
          }
LABEL_18:
          v30 = *(_QWORD *)(v26 + 24);
          v31 = *(_DWORD *)(v24 + 28);
          v32 = v30 + 8LL * *(unsigned int *)(v26 + 32);
          v33 = (_QWORD *)sub_22077B0(0x40u);
          v17 = v33;
          if ( v33 )
          {
            v33[2] = 0;
            *v33 = &v111;
            v35 = v115;
            *((_DWORD *)v17 + 6) = v31;
            v17[1] = v35;
            v115 = v17;
            *((_DWORD *)v17 + 7) = v31;
            v17[4] = v26;
            v17[5] = v30;
            v17[6] = v32;
            *((_BYTE *)v17 + 56) = 0;
          }
          v19 = (unsigned int)v117;
          v20 = v117;
          if ( HIDWORD(v117) <= (unsigned int)v117 )
          {
            v21 = (_BYTE *)sub_C8D7D0((__int64)&v116, (__int64)v118, 0, 8u, &v110, v34);
            v36 = 8LL * (unsigned int)v117;
            if ( &v21[v36] )
            {
              *(_QWORD *)&v21[v36] = v17;
              v36 = 8LL * (unsigned int)v117;
            }
            v37 = v116;
            v38 = &v116[v36];
            if ( v116 != &v116[v36] )
            {
              v39 = &v21[v36];
              v40 = v21;
              do
              {
                if ( v40 )
                {
                  *v40 = *v37;
                  *v37 = 0;
                }
                ++v40;
                ++v37;
              }
              while ( v40 != v39 );
              v41 = v116;
              v38 = &v116[8 * (unsigned int)v117];
              if ( v116 != v38 )
              {
                do
                {
                  v42 = (__int64 *)*((_QWORD *)v38 - 1);
                  v38 -= 8;
                  if ( v42 )
                  {
                    sub_2A131D0(v42);
                    j_j___libc_free_0((unsigned __int64)v42);
                  }
                }
                while ( v41 != v38 );
                goto LABEL_32;
              }
            }
            goto LABEL_33;
          }
LABEL_7:
          v21 = v116;
          v22 = &v116[8 * v19];
          if ( v22 )
          {
            *v22 = v17;
            v20 = v117;
            v21 = v116;
          }
          goto LABEL_9;
        }
        if ( sub_C8CA60(a1 + 56, v27) )
          goto LABEL_18;
      }
      v23 = v117;
      v21 = v116;
      if ( !(_DWORD)v117 )
        goto LABEL_39;
    }
    v47 = *(_DWORD *)(v24 + 24);
    v48 = (sub_AA54C0(**(_QWORD **)(v24 + 32)) == 0) + v47;
    v49 = **(_QWORD **)(v24 + 32);
    v50 = *(_QWORD *)(v49 + 56);
    v109 = v49 + 48;
    if ( v49 + 48 == v50 )
      goto LABEL_67;
    v103 = v24;
    while ( 1 )
    {
      v51 = v50;
      v50 = *(_QWORD *)(v50 + 8);
      v52 = v51 - 24;
      if ( *(_BYTE *)(v51 - 24) != 61 || sub_B46500((unsigned __int8 *)(v51 - 24)) || (*(_BYTE *)(v51 - 22) & 1) != 0 )
      {
        v48 -= ((unsigned __int8)sub_B46490(v51 - 24) == 0) - 1;
        goto LABEL_65;
      }
      v54 = sub_DD8400(a3, *(_QWORD *)(v51 - 56));
      if ( v114 )
      {
        v55 = (v114 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
        v56 = (__int64 **)(v112 + 16LL * v55);
        v57 = *v56;
        if ( v54 == *v56 )
        {
LABEL_52:
          if ( v56 != (__int64 **)(v112 + 16LL * v114) )
          {
            v58 = v56[1];
            v59 = v58[3];
            v60 = *((_DWORD *)v58 + 8);
            goto LABEL_54;
          }
        }
        else
        {
          v82 = 1;
          while ( v57 != (__int64 *)-4096LL )
          {
            v53 = (unsigned int)(v82 + 1);
            v55 = (v114 - 1) & (v82 + v55);
            v56 = (__int64 **)(v112 + 16LL * v55);
            v57 = *v56;
            if ( v54 == *v56 )
              goto LABEL_52;
            v82 = v53;
          }
        }
      }
      v60 = 0;
      v59 = 0;
LABEL_54:
      v61 = sub_2A117C0(v59, v60, v51 - 24, v48, a5, v53, a7, a8);
      if ( v61 )
      {
        if ( *(_BYTE *)v61 > 0x1Cu )
        {
          v62 = *(_QWORD *)(v61 + 40);
          v63 = *(_QWORD *)(v51 + 16);
          if ( v62 != v63 )
          {
            v64 = *(_DWORD *)(a4 + 24);
            v65 = *(_QWORD *)(a4 + 8);
            if ( v64 )
            {
              v66 = v64 - 1;
              v67 = v66 & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
              v68 = (__int64 *)(v65 + 16LL * v67);
              v69 = *v68;
              if ( *v68 == v62 )
              {
LABEL_59:
                v70 = (_QWORD *)v68[1];
                if ( v70 )
                {
                  v71 = v66 & (((unsigned int)v63 >> 9) ^ ((unsigned int)v63 >> 4));
                  v72 = (__int64 *)(v65 + 16LL * v71);
                  v73 = *v72;
                  if ( *v72 != v63 )
                  {
                    v83 = 1;
                    while ( v73 != -4096 )
                    {
                      v84 = v83 + 1;
                      v71 = v66 & (v83 + v71);
                      v72 = (__int64 *)(v65 + 16LL * v71);
                      v73 = *v72;
                      if ( v63 == *v72 )
                        goto LABEL_61;
                      v83 = v84;
                    }
                    goto LABEL_65;
                  }
LABEL_61:
                  v74 = (_QWORD *)v72[1];
                  if ( v70 != v74 )
                  {
                    while ( v74 )
                    {
                      v74 = (_QWORD *)*v74;
                      if ( v70 == v74 )
                        goto LABEL_64;
                    }
                    goto LABEL_65;
                  }
                }
              }
              else
              {
                v85 = 1;
                while ( v69 != -4096 )
                {
                  v86 = v85 + 1;
                  v67 = v66 & (v85 + v67);
                  v68 = (__int64 *)(v65 + 16LL * v67);
                  v69 = *v68;
                  if ( v62 == *v68 )
                    goto LABEL_59;
                  v85 = v86;
                }
              }
            }
          }
        }
LABEL_64:
        sub_BD84D0(v51 - 24, v61);
        sub_B43D60((_QWORD *)(v51 - 24));
        goto LABEL_65;
      }
      v105 = v115;
      if ( !v114 )
      {
        ++v111;
        goto LABEL_107;
      }
      v75 = 0;
      v76 = 1;
      v77 = (v114 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
      v78 = (__int64 **)(v112 + 16LL * v77);
      v79 = *v78;
      if ( v54 == *v78 )
        goto LABEL_71;
      while ( 1 )
      {
        if ( v79 == (__int64 *)-4096LL )
        {
          if ( !v75 )
            v75 = v78;
          ++v111;
          v93 = v113 + 1;
          if ( 4 * ((int)v113 + 1) >= 3 * v114 )
          {
LABEL_107:
            sub_2A12FF0((__int64)&v111, 2 * v114);
            if ( v114 )
            {
              v61 = 0;
              v94 = (v114 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
              v93 = v113 + 1;
              v75 = (__int64 **)(v112 + 16LL * v94);
              v95 = *v75;
              if ( v54 == *v75 )
                goto LABEL_103;
              v96 = 0;
              v97 = 1;
              while ( v95 != (__int64 *)-4096LL )
              {
                if ( !v96 && v95 == (__int64 *)-8192LL )
                  v96 = v75;
                v94 = (v114 - 1) & (v97 + v94);
                v75 = (__int64 **)(v112 + 16LL * v94);
                v95 = *v75;
                if ( v54 == *v75 )
                  goto LABEL_103;
                ++v97;
              }
LABEL_111:
              if ( v96 )
                v75 = v96;
              goto LABEL_103;
            }
          }
          else
          {
            if ( v114 - HIDWORD(v113) - v93 > v114 >> 3 )
            {
LABEL_103:
              LODWORD(v113) = v93;
              if ( *v75 != (__int64 *)-4096LL )
                --HIDWORD(v113);
              *v75 = v54;
              v80 = (__int64 *)(v75 + 1);
              v75[1] = 0;
              goto LABEL_72;
            }
            sub_2A12FF0((__int64)&v111, v114);
            if ( v114 )
            {
              v98 = 1;
              v61 = 0;
              v93 = v113 + 1;
              v96 = 0;
              v99 = (v114 - 1) & (((unsigned int)v54 >> 9) ^ ((unsigned int)v54 >> 4));
              v75 = (__int64 **)(v112 + 16 * v99);
              v100 = *v75;
              if ( v54 == *v75 )
                goto LABEL_103;
              while ( v100 != (__int64 *)-4096LL )
              {
                if ( v100 == (__int64 *)-8192LL && !v96 )
                  v96 = v75;
                LODWORD(v99) = (v114 - 1) & (v98 + v99);
                v75 = (__int64 **)(v112 + 16LL * (unsigned int)v99);
                v100 = *v75;
                if ( v54 == *v75 )
                  goto LABEL_103;
                ++v98;
              }
              goto LABEL_111;
            }
          }
          LODWORD(v113) = v113 + 1;
          BUG();
        }
        if ( v75 || v79 != (__int64 *)-8192LL )
          v78 = v75;
        v77 = (v114 - 1) & (v76 + v77);
        v79 = *(__int64 **)(v112 + 16LL * v77);
        if ( v54 == v79 )
          break;
        ++v76;
        v75 = v78;
        v78 = (__int64 **)(v112 + 16LL * v77);
      }
      v78 = (__int64 **)(v112 + 16LL * v77);
LABEL_71:
      v61 = (__int64)v78[1];
      v80 = (__int64 *)(v78 + 1);
LABEL_72:
      v101 = v61;
      v102 = v105[2];
      v81 = sub_C7D670(40, 8);
      *(_QWORD *)(v81 + 16) = v54;
      *(_QWORD *)(v81 + 24) = v52;
      *(_DWORD *)(v81 + 32) = v48;
      *(_QWORD *)v81 = v102;
      *(_QWORD *)(v81 + 8) = v101;
      *v80 = v81;
      v105[2] = v81;
LABEL_65:
      if ( v109 == v50 )
      {
        v24 = v103;
LABEL_67:
        v23 = v117;
        v21 = v116;
        *(_DWORD *)(v24 + 28) = v48;
        *(_BYTE *)(v24 + 56) = 1;
        goto LABEL_10;
      }
    }
  }
LABEL_39:
  if ( v21 != v118 )
    _libc_free((unsigned __int64)v21);
  return sub_C7D6A0(v112, 16LL * v114, 8);
}
