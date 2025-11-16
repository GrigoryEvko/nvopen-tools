// Function: sub_2E1DEA0
// Address: 0x2e1dea0
//
__int64 __fastcall sub_2E1DEA0(_QWORD *a1)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 *v4; // r13
  _QWORD *v5; // r9
  __int64 *v6; // rdi
  __int64 v7; // r12
  __int64 v8; // rdx
  unsigned __int64 *v9; // rbx
  __int64 v10; // r8
  __int64 *v11; // rdx
  __int64 v12; // r15
  __int64 v13; // r10
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // r14
  __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 **v23; // r10
  __int64 *v24; // r15
  __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r10
  __int64 *v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // r13
  __int64 **v32; // rbx
  __int64 v33; // rdx
  __int64 v34; // r11
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // rdx
  __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r8
  int v45; // r10d
  __int64 v46; // r15
  _QWORD *v47; // rdi
  __int64 v48; // r14
  __int64 v49; // r13
  __int64 v50; // r9
  unsigned int v51; // eax
  __int64 v52; // rbx
  __int64 v53; // rdx
  __int64 v54; // rax
  _QWORD *v55; // rdi
  int v56; // r12d
  _QWORD *v57; // rcx
  __int64 *v58; // rdx
  __int64 v59; // r13
  _QWORD *v60; // r11
  __int64 v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rdx
  unsigned int v65; // eax
  __int64 v66; // rax
  _QWORD *v67; // rax
  __int128 v68; // [rsp-20h] [rbp-2D0h]
  __int128 v69; // [rsp-20h] [rbp-2D0h]
  __int64 v70; // [rsp+0h] [rbp-2B0h]
  __int64 v71; // [rsp+8h] [rbp-2A8h]
  __int64 v72; // [rsp+10h] [rbp-2A0h]
  __int64 v73; // [rsp+20h] [rbp-290h]
  _QWORD *v74; // [rsp+20h] [rbp-290h]
  _QWORD *v75; // [rsp+28h] [rbp-288h]
  __int64 v76; // [rsp+28h] [rbp-288h]
  __int64 v77; // [rsp+38h] [rbp-278h]
  int v78; // [rsp+40h] [rbp-270h]
  char v79; // [rsp+47h] [rbp-269h]
  bool v80; // [rsp+48h] [rbp-268h]
  _QWORD *v81; // [rsp+48h] [rbp-268h]
  _QWORD *v82; // [rsp+48h] [rbp-268h]
  _QWORD *v83; // [rsp+50h] [rbp-260h]
  __int64 v84; // [rsp+50h] [rbp-260h]
  __int64 v85; // [rsp+50h] [rbp-260h]
  __int64 *v86; // [rsp+58h] [rbp-258h]
  _QWORD *v87; // [rsp+58h] [rbp-258h]
  unsigned int v88; // [rsp+58h] [rbp-258h]
  unsigned __int64 v89; // [rsp+58h] [rbp-258h]
  __int64 v90; // [rsp+58h] [rbp-258h]
  _QWORD *v91; // [rsp+60h] [rbp-250h]
  __int64 v92; // [rsp+60h] [rbp-250h]
  _QWORD *v93; // [rsp+60h] [rbp-250h]
  _QWORD *v94; // [rsp+60h] [rbp-250h]
  _QWORD *v95; // [rsp+60h] [rbp-250h]
  _QWORD *v96; // [rsp+60h] [rbp-250h]
  __int64 *v97; // [rsp+68h] [rbp-248h]
  _QWORD *v98; // [rsp+70h] [rbp-240h] BYREF
  __int64 v99; // [rsp+78h] [rbp-238h]
  _QWORD v100[70]; // [rsp+80h] [rbp-230h] BYREF

  do
  {
    v2 = a1[23];
    result = v2 + 32LL * *((unsigned int *)a1 + 48);
    v97 = (__int64 *)result;
    if ( v2 == result )
      return result;
    v79 = 0;
    v4 = (__int64 *)a1[23];
    v5 = a1;
    do
    {
      v60 = (_QWORD *)v4[1];
      if ( !v60 )
        goto LABEL_13;
      v20 = v60[1];
      v21 = *v60;
      result = v5[18];
      if ( !v20
        || (v22 = *(unsigned int *)(*(_QWORD *)v20 + 24LL),
            (*(_QWORD *)(v5[5] + 8LL * ((unsigned int)v22 >> 6)) & (1LL << v22)) == 0) )
      {
LABEL_3:
        v6 = (__int64 *)v5[4];
        v7 = *v4;
        v8 = 16LL * *(unsigned int *)(v21 + 24);
        v9 = (unsigned __int64 *)(result + v8);
        v10 = *(unsigned int *)(*v4 + 72);
        v11 = (__int64 *)(*(_QWORD *)(v5[2] + 152LL) + v8);
        v12 = *v11;
        v13 = v11[1];
        v14 = *v6;
        v6[10] += 16;
        v15 = (v14 + 15) & 0xFFFFFFFFFFFFFFF0LL;
        if ( v6[1] >= v15 + 16 && v14 )
        {
          *v6 = v15 + 16;
          v16 = (v14 + 15) & 0xFFFFFFFFFFFFFFF0LL;
          if ( v15 )
            goto LABEL_6;
        }
        else
        {
          v81 = v5;
          v84 = v13;
          v88 = v10;
          v94 = v60;
          v15 = sub_9D1E70((__int64)v6, 16, 16, 4);
          v5 = v81;
          v13 = v84;
          v10 = v88;
          v60 = v94;
          v16 = v15;
LABEL_6:
          *(_DWORD *)v16 = v10;
          *(_QWORD *)(v16 + 8) = v12;
        }
        v17 = *(unsigned int *)(v7 + 72);
        v18 = *(unsigned int *)(v7 + 76);
        if ( v17 + 1 > v18 )
        {
          v18 = v7 + 80;
          v82 = v5;
          v85 = v13;
          v89 = v15;
          v95 = v60;
          sub_C8D5F0(v7 + 64, (const void *)(v7 + 80), v17 + 1, 8u, v10, (__int64)v5);
          v17 = *(unsigned int *)(v7 + 72);
          v5 = v82;
          v13 = v85;
          v15 = v89;
          v60 = v95;
        }
        v19 = *(_QWORD *)(v7 + 64);
        *(_QWORD *)(v19 + 8 * v17) = v15;
        ++*(_DWORD *)(v7 + 72);
        result = v4[2];
        v4[3] = v16;
        v4[1] = 0;
        if ( (result & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( v16 )
          {
            v99 = result;
            v100[0] = v16;
            *((_QWORD *)&v68 + 1) = result;
            *(_QWORD *)&v68 = v12;
            v91 = v5;
            v98 = (_QWORD *)v12;
            result = sub_2E0F080(v7, v18, v19, v17, v10, (__int64)v5, v68, v16);
            v5 = v91;
          }
        }
        else
        {
          if ( v16 )
          {
            v99 = v13;
            v100[0] = v16;
            *((_QWORD *)&v69 + 1) = v13;
            *(_QWORD *)&v69 = v12;
            v87 = v5;
            v93 = v60;
            v98 = (_QWORD *)v12;
            result = sub_2E0F080(v7, v18, v19, v17, v10, (__int64)v5, v69, v16);
            v5 = v87;
            v60 = v93;
          }
          *v9 = v16;
          v9[1] = (unsigned __int64)v60;
        }
LABEL_12:
        v79 = 1;
        goto LABEL_13;
      }
      v23 = (__int64 **)(result + 16 * v22);
      v24 = *v23;
      v77 = (__int64)v23[1];
      v80 = v24 != 0 && v24 != &qword_501EAE0;
      if ( !v77 && *v23 != 0 && *v23 != &qword_501EAE0 )
      {
        v61 = v5[3];
        v62 = *(_QWORD *)((v24[1] & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v62 )
        {
          v63 = *(_QWORD *)(v62 + 24);
        }
        else
        {
          v66 = v5[2];
          v98 = (_QWORD *)v24[1];
          v90 = v21;
          v96 = v5;
          v67 = sub_2E1D5D0(
                  *(_QWORD **)(v66 + 296),
                  *(_QWORD *)(v66 + 296) + 16LL * *(unsigned int *)(v66 + 304),
                  (__int64 *)&v98);
          v21 = v90;
          v5 = v96;
          v63 = *(v67 - 1);
        }
        if ( v63 )
        {
          v64 = (unsigned int)(*(_DWORD *)(v63 + 24) + 1);
          v65 = *(_DWORD *)(v63 + 24) + 1;
        }
        else
        {
          v64 = 0;
          v65 = 0;
        }
        v77 = 0;
        if ( v65 < *(_DWORD *)(v61 + 32) )
          v77 = *(_QWORD *)(*(_QWORD *)(v61 + 24) + 8 * v64);
        v23[1] = (__int64 *)v77;
        v25 = *(_QWORD *)(v21 + 64);
        v26 = v25 + 8LL * *(unsigned int *)(v21 + 72);
        if ( v26 != v25 )
        {
          result = v5[18];
LABEL_20:
          v83 = v60;
          v86 = v4;
          v92 = v21;
          v27 = v26;
          while ( 1 )
          {
            v28 = result + 16LL * *(unsigned int *)(*(_QWORD *)v25 + 24LL);
            v29 = *(__int64 **)v28;
            if ( *(__int64 **)v28 == v24 || !v29 )
              goto LABEL_34;
            if ( v29 == &qword_501EAE0 )
              goto LABEL_32;
            v30 = *(_QWORD *)(v28 + 8);
            if ( v30 )
            {
              if ( v30 == v20 )
                goto LABEL_32;
            }
            else
            {
              v33 = v29[1];
              v34 = v5[3];
              v35 = *(_QWORD *)((v33 & 0xFFFFFFFFFFFFFFF8LL) + 16);
              if ( v35 )
              {
                v36 = *(_QWORD *)(v35 + 24);
              }
              else
              {
                v40 = v5[2];
                v98 = (_QWORD *)v33;
                v73 = v27;
                v75 = v5;
                v41 = sub_2E1D5D0(
                        *(_QWORD **)(v40 + 296),
                        *(_QWORD *)(v40 + 296) + 16LL * *(unsigned int *)(v40 + 304),
                        (__int64 *)&v98);
                v27 = v73;
                v5 = v75;
                v36 = *(v41 - 1);
              }
              if ( v36 )
              {
                v37 = (unsigned int)(*(_DWORD *)(v36 + 24) + 1);
                if ( (unsigned int)(*(_DWORD *)(v36 + 24) + 1) >= *(_DWORD *)(v34 + 32) )
                  goto LABEL_61;
              }
              else
              {
                v37 = 0;
                if ( !*(_DWORD *)(v34 + 32) )
                {
LABEL_61:
                  *(_QWORD *)(v28 + 8) = 0;
                  v60 = v83;
                  v21 = v92;
                  v4 = v86;
                  result = v5[18];
                  goto LABEL_3;
                }
              }
              v30 = *(_QWORD *)(*(_QWORD *)(v34 + 24) + 8 * v37);
              *(_QWORD *)(v28 + 8) = v30;
              if ( !v30 || v20 == v30 )
              {
LABEL_47:
                v60 = v83;
                v21 = v92;
                v4 = v86;
                result = v5[18];
                goto LABEL_3;
              }
            }
            if ( v20 == *(_QWORD *)(v30 + 8) )
              goto LABEL_47;
            if ( v30 == *(_QWORD *)(v20 + 8) || *(_DWORD *)(v20 + 16) >= *(_DWORD *)(v30 + 16) )
              break;
            v31 = v5[3];
            if ( *(_BYTE *)(v31 + 112) )
            {
              result = v5[18];
              if ( *(_DWORD *)(v30 + 72) >= *(_DWORD *)(v20 + 72) )
              {
LABEL_31:
                if ( *(_DWORD *)(v30 + 76) <= *(_DWORD *)(v20 + 76) )
                  goto LABEL_32;
              }
LABEL_34:
              v25 += 8;
              if ( v27 == v25 )
                goto LABEL_35;
              continue;
            }
            v38 = *(_DWORD *)(v31 + 116) + 1;
            *(_DWORD *)(v31 + 116) = v38;
            if ( v38 > 0x20 )
            {
              HIDWORD(v99) = 32;
              v98 = v100;
              v42 = *(_QWORD *)(v31 + 96);
              if ( v42 )
              {
                v43 = *(_QWORD *)(v42 + 24);
                v76 = v27;
                v44 = (__int64)v24;
                v45 = 1;
                v100[0] = *(_QWORD *)(v31 + 96);
                v46 = v20;
                v47 = v100;
                v48 = v31;
                v100[1] = v43;
                v49 = v30;
                LODWORD(v99) = 1;
                v74 = v5;
                v50 = v25;
                *(_DWORD *)(v42 + 72) = 0;
                v51 = 1;
                do
                {
                  while ( 1 )
                  {
                    v56 = v45++;
                    v57 = &v47[2 * v51 - 2];
                    v58 = (__int64 *)v57[1];
                    if ( v58 != (__int64 *)(*(_QWORD *)(*v57 + 24LL) + 8LL * *(unsigned int *)(*v57 + 32LL)) )
                      break;
                    --v51;
                    *(_DWORD *)(*v57 + 76LL) = v56;
                    LODWORD(v99) = v51;
                    if ( !v51 )
                      goto LABEL_71;
                  }
                  v52 = *v58;
                  v57[1] = v58 + 1;
                  v53 = (unsigned int)v99;
                  v54 = *(_QWORD *)(v52 + 24);
                  if ( (unsigned __int64)(unsigned int)v99 + 1 > HIDWORD(v99) )
                  {
                    v70 = v50;
                    v78 = v45;
                    v71 = v44;
                    v72 = *(_QWORD *)(v52 + 24);
                    sub_C8D5F0((__int64)&v98, v100, (unsigned int)v99 + 1LL, 0x10u, v44, v50);
                    v47 = v98;
                    v53 = (unsigned int)v99;
                    v50 = v70;
                    v45 = v78;
                    v44 = v71;
                    v54 = v72;
                  }
                  v55 = &v47[2 * v53];
                  *v55 = v52;
                  v55[1] = v54;
                  LODWORD(v99) = v99 + 1;
                  v51 = v99;
                  *(_DWORD *)(v52 + 72) = v56;
                  v47 = v98;
                }
                while ( v51 );
LABEL_71:
                v30 = v49;
                v59 = v48;
                v25 = v50;
                v20 = v46;
                *(_BYTE *)(v59 + 112) = 1;
                v24 = (__int64 *)v44;
                v5 = v74;
                *(_DWORD *)(v59 + 116) = 0;
                v27 = v76;
                if ( v47 != v100 )
                {
                  _libc_free((unsigned __int64)v47);
                  v27 = v76;
                  v5 = v74;
                }
              }
              result = v5[18];
              if ( *(_DWORD *)(v30 + 72) >= *(_DWORD *)(v20 + 72) )
                goto LABEL_31;
              v25 += 8;
              if ( v27 == v25 )
                goto LABEL_35;
            }
            else
            {
              do
              {
                v39 = v30;
                v30 = *(_QWORD *)(v30 + 8);
              }
              while ( v30 && *(_DWORD *)(v20 + 16) <= *(_DWORD *)(v30 + 16) );
              result = v5[18];
              if ( v20 == v39 )
              {
LABEL_32:
                v60 = v83;
                v21 = v92;
                v4 = v86;
                goto LABEL_3;
              }
              v25 += 8;
              if ( v27 == v25 )
              {
LABEL_35:
                v21 = v92;
                v4 = v86;
                goto LABEL_36;
              }
            }
          }
          result = v5[18];
          goto LABEL_34;
        }
        result = v5[18] + 16LL * *(unsigned int *)(v21 + 24);
        v32 = (__int64 **)result;
      }
      else
      {
        v25 = *(_QWORD *)(v21 + 64);
        v26 = v25 + 8LL * *(unsigned int *)(v21 + 72);
        if ( v25 != v26 )
          goto LABEL_20;
LABEL_36:
        if ( !v80 )
          goto LABEL_13;
        v32 = (__int64 **)(result + 16LL * *(unsigned int *)(v21 + 24));
      }
      v4[3] = (__int64)v24;
      if ( (v4[2] & 0xFFFFFFFFFFFFFFF8LL) == 0 && *v32 != v24 )
      {
        result = v77;
        *v32 = v24;
        v32[1] = (__int64 *)v77;
        goto LABEL_12;
      }
LABEL_13:
      v4 += 4;
    }
    while ( v97 != v4 );
    a1 = v5;
  }
  while ( v79 );
  return result;
}
