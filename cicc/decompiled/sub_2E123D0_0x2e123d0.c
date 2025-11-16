// Function: sub_2E123D0
// Address: 0x2e123d0
//
void __fastcall sub_2E123D0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, __int64 a6)
{
  int v6; // r10d
  unsigned __int64 v11; // rcx
  unsigned int v12; // eax
  __int64 v13; // rbx
  __int64 *v14; // rdx
  __int64 *v15; // rbx
  unsigned int v16; // eax
  unsigned __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // r15
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 *v23; // rdx
  __int64 v24; // r8
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // r13
  __int64 v31; // r10
  unsigned __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rcx
  _QWORD *v35; // r11
  unsigned int v36; // edx
  __int64 v37; // rdi
  __int64 *v38; // rsi
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 *v42; // rax
  unsigned __int64 v43; // rcx
  __int64 v44; // r8
  __int64 v45; // rax
  __int64 *v46; // r15
  __int64 *v47; // rbx
  __int64 v48; // r13
  __int64 *v49; // rax
  __int64 *v50; // rdx
  unsigned __int64 v51; // rcx
  __int64 v52; // r8
  __int64 *v53; // r9
  __int64 *v54; // r13
  __int64 v55; // rbx
  __int64 *v56; // rax
  __int64 v57; // r10
  signed __int64 v58; // rbx
  __int64 v59; // r10
  __int64 v60; // rax
  __int64 *v61; // rax
  unsigned __int64 v62; // rdx
  __int64 v63; // r9
  signed __int64 v64; // r13
  __int64 v65; // r13
  __int64 v66; // rax
  __int64 *v67; // rax
  __int64 v68; // rax
  __int64 v69; // r8
  _QWORD *v70; // rdx
  _QWORD *v71; // rdi
  __int128 v72; // [rsp-20h] [rbp-1A0h]
  const void *v73; // [rsp+8h] [rbp-178h]
  __int64 v74; // [rsp+8h] [rbp-178h]
  __int64 v75; // [rsp+10h] [rbp-170h]
  int v76; // [rsp+10h] [rbp-170h]
  __int64 v78; // [rsp+20h] [rbp-160h]
  __int64 v79; // [rsp+20h] [rbp-160h]
  __int64 v80; // [rsp+20h] [rbp-160h]
  __int64 v81; // [rsp+20h] [rbp-160h]
  __int64 v82; // [rsp+20h] [rbp-160h]
  __int64 v83; // [rsp+28h] [rbp-158h]
  __int64 *v84; // [rsp+28h] [rbp-158h]
  __int64 *v85; // [rsp+28h] [rbp-158h]
  __int64 v86; // [rsp+28h] [rbp-158h]
  __int64 v87; // [rsp+50h] [rbp-130h] BYREF
  __int64 *v88; // [rsp+58h] [rbp-128h]
  __int64 v89; // [rsp+60h] [rbp-120h]
  int v90; // [rsp+68h] [rbp-118h]
  char v91; // [rsp+6Ch] [rbp-114h]
  char v92; // [rsp+70h] [rbp-110h] BYREF
  __int64 v93; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 *v94; // [rsp+B8h] [rbp-C8h]
  __int64 v95; // [rsp+C0h] [rbp-C0h]
  int v96; // [rsp+C8h] [rbp-B8h]
  char v97; // [rsp+CCh] [rbp-B4h]
  char v98; // [rsp+D0h] [rbp-B0h] BYREF

  v6 = a4;
  v88 = (__int64 *)&v92;
  v94 = (__int64 *)&v98;
  v11 = *(unsigned int *)(a1 + 160);
  v12 = a4 & 0x7FFFFFFF;
  v87 = 0;
  v91 = 1;
  v13 = 8LL * v12;
  v89 = 8;
  v90 = 0;
  v93 = 0;
  v95 = 16;
  v96 = 0;
  v97 = 1;
  if ( v12 >= (unsigned int)v11 || (v14 = *(__int64 **)(*(_QWORD *)(a1 + 152) + 8LL * v12)) == 0 )
  {
    v39 = v12 + 1;
    if ( (unsigned int)v11 < v39 )
    {
      v62 = v39;
      if ( v39 != v11 )
      {
        if ( v39 >= v11 )
        {
          v68 = *(_QWORD *)(a1 + 168);
          v69 = v62 - v11;
          if ( v62 > *(unsigned int *)(a1 + 164) )
          {
            v74 = a6;
            v76 = v6;
            v82 = v62 - v11;
            v86 = *(_QWORD *)(a1 + 168);
            sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v62, 8u, v69, a6);
            v11 = *(unsigned int *)(a1 + 160);
            a6 = v74;
            v6 = v76;
            v69 = v82;
            v68 = v86;
          }
          v40 = *(_QWORD *)(a1 + 152);
          v70 = (_QWORD *)(v40 + 8 * v11);
          v71 = &v70[v69];
          if ( v70 != v71 )
          {
            do
              *v70++ = v68;
            while ( v71 != v70 );
            LODWORD(v11) = *(_DWORD *)(a1 + 160);
            v40 = *(_QWORD *)(a1 + 152);
          }
          *(_DWORD *)(a1 + 160) = v69 + v11;
          goto LABEL_23;
        }
        *(_DWORD *)(a1 + 160) = v39;
      }
    }
    v40 = *(_QWORD *)(a1 + 152);
LABEL_23:
    v79 = a6;
    v41 = sub_2E10F30(v6);
    *(_QWORD *)(v40 + v13) = v41;
    v83 = v41;
    sub_2E11E80((_QWORD *)a1, v41);
    a6 = v79;
    v14 = (__int64 *)v83;
  }
  if ( a6 | a5 )
  {
    v15 = (__int64 *)v14[13];
    if ( !v15 )
LABEL_93:
      BUG();
    while ( !(a5 & v15[14] | a6 & v15[15]) )
    {
      v15 = (__int64 *)v15[13];
      if ( !v15 )
        goto LABEL_93;
    }
  }
  else
  {
    v15 = v14;
  }
  v16 = *(_DWORD *)(a3 + 8);
  if ( v16 )
  {
    v73 = (const void *)(a3 + 16);
    while ( 1 )
    {
      v28 = (__int64 *)(*(_QWORD *)a3 + 16LL * v16 - 16);
      v29 = *v28;
      v30 = v28[1];
      *(_DWORD *)(a3 + 8) = v16 - 1;
      v31 = *(_QWORD *)(a1 + 32);
      v32 = v29 & 0xFFFFFFFFFFFFFFF8LL;
      v33 = (v29 >> 1) & 3;
      if ( ((v29 >> 1) & 3) != 0 )
      {
        v19 = v32 | (2LL * ((int)v33 - 1));
        v17 = v29 & 0xFFFFFFFFFFFFFFF8LL | (2LL * ((int)v33 - 1)) & 0xFFFFFFFFFFFFFFF8LL;
        v18 = *(_QWORD *)(v17 + 16);
        if ( !v18 )
        {
LABEL_15:
          v34 = *(unsigned int *)(v31 + 304);
          v35 = *(_QWORD **)(v31 + 296);
          if ( *(_DWORD *)(v31 + 304) )
          {
            v36 = *(_DWORD *)(v17 + 24) | (v19 >> 1) & 3;
            do
            {
              while ( 1 )
              {
                v37 = v34 >> 1;
                v38 = &v35[2 * (v34 >> 1)];
                if ( v36 < (*(_DWORD *)((*v38 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v38 >> 1) & 3) )
                  break;
                v35 = v38 + 2;
                v34 = v34 - v37 - 1;
                if ( v34 <= 0 )
                  goto LABEL_20;
              }
              v34 >>= 1;
            }
            while ( v37 > 0 );
          }
LABEL_20:
          v20 = *(v35 - 1);
          goto LABEL_9;
        }
      }
      else
      {
        v17 = *(_QWORD *)v32 & 0xFFFFFFFFFFFFFFF8LL;
        v18 = *(_QWORD *)(v17 + 16);
        v19 = v17 | 6;
        if ( !v18 )
          goto LABEL_15;
      }
      v20 = *(_QWORD *)(v18 + 24);
LABEL_9:
      v78 = v29;
      v21 = *(_QWORD *)(*(_QWORD *)(v31 + 152) + 16LL * *(unsigned int *)(v20 + 24));
      v22 = sub_2E0EAB0(a2, v21, v29);
      v25 = v21;
      v26 = v78;
      if ( v22 )
      {
        v27 = *(_QWORD *)(v30 + 8);
        if ( (v27 & 6) != 0 || v21 != v27 )
          goto LABEL_12;
        if ( !v91 )
          goto LABEL_42;
        v42 = v88;
        v25 = HIDWORD(v89);
        v23 = &v88[HIDWORD(v89)];
        if ( v88 != v23 )
        {
          while ( v30 != *v42 )
          {
            if ( v23 == ++v42 )
              goto LABEL_28;
          }
          goto LABEL_12;
        }
LABEL_28:
        if ( HIDWORD(v89) < (unsigned int)v89 )
        {
          v43 = (unsigned int)++HIDWORD(v89);
          *v23 = v30;
          ++v87;
        }
        else
        {
LABEL_42:
          sub_C8CC70((__int64)&v87, v30, (__int64)v23, v25, v24, v78);
          if ( !(_BYTE)v23 )
            goto LABEL_12;
        }
        v44 = *(_QWORD *)(v20 + 64);
        v45 = *(unsigned int *)(v20 + 72);
        if ( v44 != v44 + 8 * v45 )
        {
          v84 = (__int64 *)(v44 + 8 * v45);
          v46 = v15;
          v47 = (__int64 *)v44;
          while ( 2 )
          {
            while ( 2 )
            {
              v48 = *v47;
              if ( v97 )
              {
                v49 = v94;
                v43 = HIDWORD(v95);
                v23 = &v94[HIDWORD(v95)];
                if ( v94 != v23 )
                {
                  while ( v48 != *v49 )
                  {
                    if ( v23 == ++v49 )
                      goto LABEL_82;
                  }
                  break;
                }
LABEL_82:
                if ( HIDWORD(v95) < (unsigned int)v95 )
                {
                  ++HIDWORD(v95);
                  *v23 = v48;
                  ++v93;
                  goto LABEL_73;
                }
              }
              sub_C8CC70((__int64)&v93, *v47, (__int64)v23, v43, v44, v26);
              if ( (_BYTE)v23 )
              {
LABEL_73:
                v63 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 152LL) + 16LL * *(unsigned int *)(v48 + 24) + 8);
                if ( ((v63 >> 1) & 3) != 0 )
                  v64 = v63 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v63 >> 1) & 3) - 1));
                else
                  v64 = *(_QWORD *)(v63 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
                v81 = v63;
                v23 = (__int64 *)sub_2E09D00(v46, v64);
                v43 = 3LL * *((unsigned int *)v46 + 2);
                if ( v23 != (__int64 *)(*v46 + 24LL * *((unsigned int *)v46 + 2)) )
                {
                  v26 = v81;
                  v43 = v64 & 0xFFFFFFFFFFFFFFF8LL;
                  if ( (*(_DWORD *)((*v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v23 >> 1) & 3) <= (*(_DWORD *)((v64 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v64 >> 1) & 3) )
                  {
                    v65 = v23[2];
                    if ( v65 )
                    {
                      v66 = *(unsigned int *)(a3 + 8);
                      v43 = *(unsigned int *)(a3 + 12);
                      v23 = (__int64 *)(v66 + 1);
                      if ( v66 + 1 > v43 )
                      {
                        sub_C8D5F0(a3, v73, (unsigned __int64)v23, 0x10u, v44, v81);
                        v66 = *(unsigned int *)(a3 + 8);
                        v26 = v81;
                      }
                      v67 = (__int64 *)(*(_QWORD *)a3 + 16 * v66);
                      ++v47;
                      *v67 = v26;
                      v67[1] = v65;
                      ++*(_DWORD *)(a3 + 8);
                      if ( v84 == v47 )
                        goto LABEL_38;
                      continue;
                    }
                  }
                }
              }
              break;
            }
            if ( v84 == ++v47 )
              goto LABEL_38;
            continue;
          }
        }
LABEL_12:
        v16 = *(_DWORD *)(a3 + 8);
        if ( !v16 )
          break;
      }
      else
      {
        *((_QWORD *)&v72 + 1) = v78;
        *(_QWORD *)&v72 = v21;
        sub_2E0F080(a2, v21, (__int64)v23, v21, v24, v78, v72, v30);
        v53 = *(__int64 **)(v20 + 64);
        v85 = &v53[*(unsigned int *)(v20 + 72)];
        if ( v53 == v85 )
          goto LABEL_12;
        v75 = v30;
        v54 = *(__int64 **)(v20 + 64);
        v46 = v15;
        v55 = *v53;
        if ( v97 )
        {
LABEL_50:
          v56 = v94;
          v51 = HIDWORD(v95);
          v50 = &v94[HIDWORD(v95)];
          if ( v94 == v50 )
            goto LABEL_66;
          while ( v55 != *v56 )
          {
            if ( v50 == ++v56 )
            {
LABEL_66:
              if ( HIDWORD(v95) < (unsigned int)v95 )
              {
                ++HIDWORD(v95);
                *v50 = v55;
                ++v93;
                goto LABEL_57;
              }
              goto LABEL_56;
            }
          }
          goto LABEL_54;
        }
        while ( 1 )
        {
LABEL_56:
          sub_C8CC70((__int64)&v93, v55, (__int64)v50, v51, v52, (__int64)v53);
          if ( !(_BYTE)v50 )
          {
LABEL_54:
            if ( v85 == ++v54 )
              break;
            goto LABEL_55;
          }
LABEL_57:
          v57 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 152LL) + 16LL * *(unsigned int *)(v55 + 24) + 8);
          if ( ((v57 >> 1) & 3) != 0 )
            v58 = v57 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v57 >> 1) & 3) - 1));
          else
            v58 = *(_QWORD *)(v57 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
          v80 = v57;
          v50 = (__int64 *)sub_2E09D00(v46, v58);
          v51 = 3LL * *((unsigned int *)v46 + 2);
          if ( v50 == (__int64 *)(*v46 + 24LL * *((unsigned int *)v46 + 2)) )
            goto LABEL_54;
          v59 = v80;
          v51 = v58 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_DWORD *)((*v50 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v50 >> 1) & 3) > (*(_DWORD *)((v58 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                                  | (unsigned int)(v58 >> 1) & 3)
            || !v50[2] )
          {
            goto LABEL_54;
          }
          v60 = *(unsigned int *)(a3 + 8);
          v51 = *(unsigned int *)(a3 + 12);
          v50 = (__int64 *)(v60 + 1);
          if ( v60 + 1 > v51 )
          {
            sub_C8D5F0(a3, v73, (unsigned __int64)v50, 0x10u, v52, (__int64)v53);
            v60 = *(unsigned int *)(a3 + 8);
            v59 = v80;
          }
          v61 = (__int64 *)(*(_QWORD *)a3 + 16 * v60);
          ++v54;
          *v61 = v59;
          v61[1] = v75;
          ++*(_DWORD *)(a3 + 8);
          if ( v85 == v54 )
            break;
LABEL_55:
          v55 = *v54;
          if ( v97 )
            goto LABEL_50;
        }
LABEL_38:
        v16 = *(_DWORD *)(a3 + 8);
        v15 = v46;
        if ( !v16 )
          break;
      }
    }
  }
  if ( v97 )
  {
    if ( v91 )
      return;
LABEL_86:
    _libc_free((unsigned __int64)v88);
    return;
  }
  _libc_free((unsigned __int64)v94);
  if ( !v91 )
    goto LABEL_86;
}
