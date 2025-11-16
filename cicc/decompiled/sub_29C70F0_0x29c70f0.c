// Function: sub_29C70F0
// Address: 0x29c70f0
//
__int64 __fastcall sub_29C70F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, const void *a5, size_t a6)
{
  __int64 v9; // r13
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rbx
  unsigned int v13; // ecx
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // r14
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // rbx
  unsigned int v24; // esi
  __int64 v25; // r15
  __int64 v26; // r9
  int v27; // r11d
  __int64 *v28; // rcx
  unsigned int v29; // r8d
  __int64 *v30; // rax
  __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  char v36; // al
  char v37; // dl
  __int64 v38; // rdi
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // r14
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  _DWORD *v49; // rax
  int v50; // r11d
  int v51; // r11d
  __int64 v52; // r10
  unsigned int v53; // edx
  int v54; // eax
  __int64 v55; // r8
  int v56; // edi
  __int64 *v57; // rsi
  int v58; // eax
  __int64 v59; // rax
  int v60; // edx
  __int64 *v61; // rax
  __int64 v62; // rdx
  _QWORD *v63; // rdx
  bool v64; // zf
  __int64 v65; // rax
  __int64 v66; // r9
  _QWORD *v67; // r15
  __int64 *v68; // rax
  __int64 v69; // rdx
  _QWORD *v70; // rdx
  int v71; // eax
  unsigned __int64 v72; // rdi
  int v73; // r11d
  int v74; // r11d
  __int64 v75; // r10
  int v76; // edi
  unsigned int v77; // edx
  __int64 v78; // r8
  int v79; // eax
  int v80; // r9d
  unsigned int v81; // ecx
  _QWORD *v82; // r14
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // r9
  __int64 v88; // rdx
  _DWORD *v89; // rax
  unsigned __int8 v90; // al
  __int64 v91; // rdx
  __int64 v92; // rdx
  unsigned __int8 v93; // al
  __int64 v94; // rcx
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r14
  _QWORD *v98; // r15
  __int64 v99; // r13
  _QWORD *v100; // rbx
  __int64 *v102; // rax
  __int64 v103; // rax
  __int64 v104; // [rsp+8h] [rbp-B8h]
  __int64 v105; // [rsp+10h] [rbp-B0h]
  __int64 v107; // [rsp+20h] [rbp-A0h]
  __int64 v108; // [rsp+28h] [rbp-98h]
  _QWORD *v109; // [rsp+38h] [rbp-88h]
  __int64 v110; // [rsp+50h] [rbp-70h]
  int v111; // [rsp+50h] [rbp-70h]
  unsigned int v112; // [rsp+50h] [rbp-70h]
  __int64 v113; // [rsp+50h] [rbp-70h]
  __int64 v114; // [rsp+58h] [rbp-68h]
  unsigned __int8 *v115; // [rsp+68h] [rbp-58h] BYREF
  _QWORD *v116; // [rsp+70h] [rbp-50h] BYREF
  __int64 v117[2]; // [rsp+78h] [rbp-48h] BYREF
  _QWORD *v118; // [rsp+88h] [rbp-38h]

  if ( sub_BA8DC0(a1, (__int64)"llvm.dbg.cu", 11) )
  {
    v108 = a2;
    v9 = a4;
    v105 = *(unsigned int *)(a4 + 40);
    v104 = a4 + 96;
    if ( a3 == a2 )
      return 1;
    while ( 1 )
    {
      v10 = *(unsigned int *)(v9 + 24);
      v11 = *(_QWORD *)(v9 + 8);
      v12 = v108 - 56;
      if ( !v108 )
        v12 = 0;
      if ( (_DWORD)v10 )
      {
        v13 = (v10 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
        v14 = (__int64 *)(v11 + 16LL * v13);
        v15 = *v14;
        if ( v12 == *v14 )
        {
LABEL_9:
          if ( v14 != (__int64 *)(v11 + 16 * v10) )
            goto LABEL_4;
        }
        else
        {
          v79 = 1;
          while ( v15 != -4096 )
          {
            v80 = v79 + 1;
            v13 = (v10 - 1) & (v79 + v13);
            v14 = (__int64 *)(v11 + 16LL * v13);
            v15 = *v14;
            if ( v12 == *v14 )
              goto LABEL_9;
            v79 = v80;
          }
        }
      }
      if ( !sub_B2FC80(v12) && !sub_B2FC80(v12) && !(unsigned __int8)sub_B2FC00((_BYTE *)v12) )
      {
        if ( ++v105 >= (unsigned __int64)qword_5008EE8 )
          return 1;
        v16 = sub_B92180(v12);
        v116 = (_QWORD *)v12;
        v17 = v16;
        v114 = v16;
        v117[0] = v16;
        sub_29C6AD0(v9, (__int64 *)&v116, v117, v18, v19, v20);
        if ( v17 )
        {
          v90 = *(_BYTE *)(v114 - 16);
          v91 = (v90 & 2) != 0 ? *(_QWORD *)(v114 - 32) : v114 - 16 - 8LL * ((v90 >> 2) & 0xF);
          v92 = *(_QWORD *)(v91 + 56);
          if ( v92 )
          {
            v93 = *(_BYTE *)(v92 - 16);
            if ( (v93 & 2) != 0 )
            {
              v94 = *(_QWORD *)(v92 - 32);
              v95 = *(unsigned int *)(v92 - 24);
            }
            else
            {
              v94 = v92 - 16 - 8LL * ((v93 >> 2) & 0xF);
              v95 = (*(_WORD *)(v92 - 16) >> 6) & 0xF;
            }
            v96 = v94 + 8 * v95;
            if ( v94 != v96 )
            {
              v113 = v9;
              v97 = v9 + 144;
              v98 = (_QWORD *)v94;
              v99 = v12;
              v100 = (_QWORD *)(v94 + 8 * v95);
              do
              {
                if ( !*v98 )
                  BUG();
                if ( *(_BYTE *)*v98 == 26 )
                {
                  v116 = (_QWORD *)*v98;
                  *(_DWORD *)sub_29C5270(v97, (__int64 *)&v116, v96, v94, v21, v22) = 0;
                }
                ++v98;
              }
              while ( v100 != v98 );
              v12 = v99;
              v9 = v113;
            }
          }
        }
        v107 = v12 + 72;
        v109 = *(_QWORD **)(v12 + 80);
        if ( (_QWORD *)(v12 + 72) != v109 )
        {
          while ( 1 )
          {
            if ( !v109 )
              BUG();
            v23 = (_QWORD *)v109[4];
            if ( v109 + 3 != v23 )
              break;
LABEL_73:
            v109 = (_QWORD *)v109[1];
            if ( (_QWORD *)v107 == v109 )
              goto LABEL_4;
          }
          while ( 2 )
          {
            if ( !v23 )
              BUG();
            v36 = *((_BYTE *)v23 - 24);
            v37 = v36;
            if ( v36 == 84 )
              goto LABEL_28;
            if ( (int)qword_5008C88 > 0 )
            {
              v38 = v23[5];
              if ( v38 )
              {
                v40 = sub_B14240(v38);
                v41 = v39;
                if ( v39 != v40 )
                {
                  while ( *(_BYTE *)(v40 + 32) )
                  {
                    v40 = *(_QWORD *)(v40 + 8);
                    if ( v39 == v40 )
                      goto LABEL_47;
                  }
                  if ( v39 != v40 )
                  {
LABEL_38:
                    if ( v114 )
                    {
                      v42 = *(_QWORD *)(v40 + 24);
                      v116 = (_QWORD *)v42;
                      if ( v42 )
                        sub_B96E90((__int64)&v116, v42, 1);
                      v43 = sub_B10D40((__int64)&v116);
                      if ( v116 )
                      {
                        v110 = v43;
                        sub_B91220((__int64)&v116, (__int64)v116);
                        v43 = v110;
                      }
                      if ( !v43 && !sub_B12EE0(v40) )
                      {
                        v116 = (_QWORD *)sub_B12000(v40 + 72);
                        v49 = (_DWORD *)sub_29C5270(v9 + 144, (__int64 *)&v116, v45, v46, v47, v48);
                        ++*v49;
                      }
                    }
                    while ( 1 )
                    {
                      v40 = *(_QWORD *)(v40 + 8);
                      if ( v41 == v40 )
                        break;
                      if ( !*(_BYTE *)(v40 + 32) )
                      {
                        if ( v41 != v40 )
                          goto LABEL_38;
                        break;
                      }
                    }
                  }
                }
LABEL_47:
                v36 = *((_BYTE *)v23 - 24);
              }
              if ( v36 != 85 )
                goto LABEL_20;
              v44 = *(v23 - 7);
              if ( !v44 )
                goto LABEL_20;
              if ( *(_BYTE *)v44 || *(_QWORD *)(v44 + 24) != v23[7] || (*(_BYTE *)(v44 + 33) & 0x20) == 0 )
                goto LABEL_52;
              v81 = *(_DWORD *)(v44 + 36);
              if ( v81 <= 0x45 )
              {
                if ( v81 > 0x43 )
                  goto LABEL_113;
LABEL_52:
                if ( !v44
                  || *(_BYTE *)v44
                  || *(_QWORD *)(v44 + 24) != v23[7]
                  || (*(_BYTE *)(v44 + 33) & 0x20) == 0
                  || (unsigned int)(*(_DWORD *)(v44 + 36) - 68) > 3 )
                {
LABEL_20:
                  v117[0] = 4;
                  v116 = v23 - 3;
                  v117[1] = 0;
                  v118 = v23 - 3;
                  if ( v23 != (_QWORD *)-4072LL && v23 != (_QWORD *)-8168LL )
                  {
                    sub_BD73F0((__int64)v117);
                    v24 = *(_DWORD *)(v9 + 120);
                    v25 = (__int64)v116;
                    if ( v24 )
                      goto LABEL_23;
LABEL_63:
                    ++*(_QWORD *)(v9 + 96);
LABEL_64:
                    sub_A41E30(v104, 2 * v24);
                    v50 = *(_DWORD *)(v9 + 120);
                    if ( !v50 )
                      goto LABEL_152;
                    v51 = v50 - 1;
                    v52 = *(_QWORD *)(v9 + 104);
                    v53 = v51 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
                    v54 = *(_DWORD *)(v9 + 112) + 1;
                    v28 = (__int64 *)(v52 + 16LL * v53);
                    v55 = *v28;
                    if ( *v28 != v25 )
                    {
                      v56 = 1;
                      v57 = 0;
                      while ( v55 != -4096 )
                      {
                        if ( !v57 && v55 == -8192 )
                          v57 = v28;
                        v53 = v51 & (v56 + v53);
                        v28 = (__int64 *)(v52 + 16LL * v53);
                        v55 = *v28;
                        if ( *v28 == v25 )
                          goto LABEL_84;
                        ++v56;
                      }
LABEL_103:
                      if ( v57 )
                        v28 = v57;
                    }
                    goto LABEL_84;
                  }
                  v24 = *(_DWORD *)(v9 + 120);
                  v25 = (__int64)(v23 - 3);
                  if ( !v24 )
                    goto LABEL_63;
LABEL_23:
                  v26 = *(_QWORD *)(v9 + 104);
                  v27 = 1;
                  v28 = 0;
                  v29 = (v24 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
                  v30 = (__int64 *)(v26 + 16LL * v29);
                  v31 = *v30;
                  if ( *v30 != v25 )
                  {
                    while ( v31 != -4096 )
                    {
                      if ( v31 != -8192 || v28 )
                        v30 = v28;
                      v29 = (v24 - 1) & (v27 + v29);
                      v31 = *(_QWORD *)(v26 + 16LL * v29);
                      if ( v31 == v25 )
                        goto LABEL_24;
                      ++v27;
                      v28 = v30;
                      v30 = (__int64 *)(v26 + 16LL * v29);
                    }
                    if ( !v28 )
                      v28 = v30;
                    v58 = *(_DWORD *)(v9 + 112);
                    ++*(_QWORD *)(v9 + 96);
                    v54 = v58 + 1;
                    if ( 4 * v54 >= 3 * v24 )
                      goto LABEL_64;
                    if ( v24 - *(_DWORD *)(v9 + 116) - v54 <= v24 >> 3 )
                    {
                      v112 = ((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4);
                      sub_A41E30(v104, v24);
                      v73 = *(_DWORD *)(v9 + 120);
                      if ( !v73 )
                      {
LABEL_152:
                        ++*(_DWORD *)(v9 + 112);
                        BUG();
                      }
                      v74 = v73 - 1;
                      v75 = *(_QWORD *)(v9 + 104);
                      v76 = 1;
                      v57 = 0;
                      v77 = v74 & v112;
                      v54 = *(_DWORD *)(v9 + 112) + 1;
                      v28 = (__int64 *)(v75 + 16LL * (v74 & v112));
                      v78 = *v28;
                      if ( *v28 != v25 )
                      {
                        while ( v78 != -4096 )
                        {
                          if ( !v57 && v78 == -8192 )
                            v57 = v28;
                          v77 = v74 & (v76 + v77);
                          v28 = (__int64 *)(v75 + 16LL * v77);
                          v78 = *v28;
                          if ( *v28 == v25 )
                            goto LABEL_84;
                          ++v76;
                        }
                        goto LABEL_103;
                      }
                    }
LABEL_84:
                    *(_DWORD *)(v9 + 112) = v54;
                    if ( *v28 != -4096 )
                      --*(_DWORD *)(v9 + 116);
                    *((_DWORD *)v28 + 2) = 0;
                    *v28 = v25;
                    *((_DWORD *)v28 + 2) = *(_DWORD *)(v9 + 136);
                    v59 = *(unsigned int *)(v9 + 136);
                    v60 = v59;
                    if ( *(_DWORD *)(v9 + 140) <= (unsigned int)v59 )
                    {
                      v65 = sub_C8D7D0(v9 + 128, v9 + 144, 0, 0x20u, (unsigned __int64 *)&v115, v9 + 128);
                      v66 = v9 + 128;
                      v67 = (_QWORD *)v65;
                      v68 = (__int64 *)(v65 + 32LL * *(unsigned int *)(v9 + 136));
                      if ( v68 )
                      {
                        v69 = (__int64)v116;
                        v68[1] = 4;
                        v68[2] = 0;
                        *v68 = v69;
                        v70 = v118;
                        v64 = v118 == 0;
                        v68[3] = (__int64)v118;
                        if ( v70 + 512 != 0 && !v64 && v70 != (_QWORD *)-8192LL )
                        {
                          sub_BD6050((unsigned __int64 *)v68 + 1, v117[0] & 0xFFFFFFFFFFFFFFF8LL);
                          v66 = v9 + 128;
                        }
                      }
                      sub_29C2140(v66, v67);
                      v71 = (int)v115;
                      v72 = *(_QWORD *)(v9 + 128);
                      if ( v9 + 144 != v72 )
                      {
                        v111 = (int)v115;
                        _libc_free(v72);
                        v71 = v111;
                      }
                      ++*(_DWORD *)(v9 + 136);
                      *(_QWORD *)(v9 + 128) = v67;
                      *(_DWORD *)(v9 + 140) = v71;
                    }
                    else
                    {
                      v61 = (__int64 *)(*(_QWORD *)(v9 + 128) + 32 * v59);
                      if ( v61 )
                      {
                        v62 = (__int64)v116;
                        v61[1] = 4;
                        v61[2] = 0;
                        *v61 = v62;
                        v63 = v118;
                        v64 = v118 + 512 == 0;
                        v61[3] = (__int64)v118;
                        if ( v63 != 0 && !v64 && v63 != (_QWORD *)-8192LL )
                          sub_BD6050((unsigned __int64 *)v61 + 1, v117[0] & 0xFFFFFFFFFFFFFFF8LL);
                        v60 = *(_DWORD *)(v9 + 136);
                      }
                      *(_DWORD *)(v9 + 136) = v60 + 1;
                    }
                  }
LABEL_24:
                  if ( v118 + 512 != 0 && v118 != 0 && v118 != (_QWORD *)-8192LL )
                    sub_BD60C0(v117);
                  v32 = sub_B10CD0((__int64)(v23 + 3));
                  v116 = v23 - 3;
                  LOBYTE(v117[0]) = v32 != 0;
                  sub_29C6DE0(v9 + 48, (__int64 *)&v116, v117, v33, v34, v35);
                }
LABEL_28:
                v23 = (_QWORD *)v23[1];
                if ( v109 + 3 == v23 )
                  goto LABEL_73;
                continue;
              }
              if ( v81 != 71 )
                goto LABEL_52;
LABEL_113:
              if ( !v114 )
                goto LABEL_116;
              v82 = v23 - 3;
              if ( sub_B10D40((__int64)(v23 + 3)) )
                goto LABEL_115;
              v83 = *((_DWORD *)v23 - 5) & 0x7FFFFFF;
              v84 = *(_QWORD *)(v23[-4 * v83 - 3] + 24LL);
              v115 = (unsigned __int8 *)v84;
              if ( *(_BYTE *)v84 != 4 )
              {
                if ( (unsigned __int8)(*(_BYTE *)v84 - 5) > 0x1Fu )
                  goto LABEL_122;
LABEL_115:
                v36 = *((_BYTE *)v23 - 24);
LABEL_116:
                if ( v36 != 85 )
                  goto LABEL_20;
LABEL_117:
                v44 = *(v23 - 7);
                goto LABEL_52;
              }
              if ( !*(_DWORD *)(v84 + 144) && !(unsigned __int8)sub_AF4500(*(_QWORD *)(v82[4 * (2 - v83)] + 24LL)) )
                goto LABEL_115;
LABEL_122:
              sub_B58DC0(&v116, &v115);
              if ( sub_29C12C0((__int64 *)&v116) )
                goto LABEL_115;
              v88 = *((_DWORD *)v23 - 5) & 0x7FFFFFF;
              v116 = *(_QWORD **)(v82[4 * (1 - v88)] + 24LL);
              v89 = (_DWORD *)sub_29C5270(v9 + 144, (__int64 *)&v116, v88, v85, v86, v87);
              ++*v89;
              v37 = *((_BYTE *)v23 - 24);
            }
            break;
          }
          if ( v37 != 85 )
            goto LABEL_20;
          goto LABEL_117;
        }
      }
LABEL_4:
      v108 = *(_QWORD *)(v108 + 8);
      if ( a3 == v108 )
        return 1;
    }
  }
  v102 = sub_29C0AE0();
  v103 = sub_A51340((__int64)v102, a5, a6);
  sub_904010(v103, ": Skipping module without debug info\n");
  return 0;
}
