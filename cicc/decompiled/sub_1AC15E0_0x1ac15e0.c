// Function: sub_1AC15E0
// Address: 0x1ac15e0
//
__int64 *__fastcall sub_1AC15E0(_QWORD *a1)
{
  __int64 *result; // rax
  _QWORD *v2; // rbx
  unsigned __int64 v3; // r13
  __int64 v4; // rdx
  int v5; // r9d
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  unsigned int v10; // r12d
  unsigned int v11; // esi
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rdx
  char *v15; // rdi
  _BYTE *v16; // rax
  _BYTE *v17; // r10
  int v18; // edx
  size_t v19; // r11
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rsi
  unsigned int v24; // r8d
  _QWORD *v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rbx
  __int64 *v28; // rax
  int v29; // r8d
  __int64 v30; // rcx
  __int64 *v31; // r14
  int v32; // eax
  _BYTE *v33; // rsi
  __int64 v34; // rdi
  unsigned int v35; // esi
  int v36; // r9d
  __int64 v37; // r11
  unsigned int v38; // eax
  __int64 *v39; // rbx
  __int64 v40; // rdx
  __int64 v41; // r15
  __int64 v42; // rdi
  const char *v43; // rbx
  __int64 v44; // rcx
  __int64 *v45; // r12
  char *v46; // rdx
  char *v47; // rdi
  __int64 v48; // rax
  __int64 v49; // rsi
  char *v50; // rax
  _BYTE *v51; // rsi
  const char **v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rdx
  const char *v55; // rdx
  const char **v56; // r14
  const char **v57; // r13
  const char *v58; // r12
  int v59; // eax
  __int64 v60; // r11
  __int64 v61; // rdx
  int v62; // eax
  __int64 v63; // rdi
  int v64; // eax
  __int64 *v65; // rax
  int v66; // eax
  int v67; // r8d
  __int64 v68; // rax
  __int64 *v69; // r10
  int v70; // eax
  int v71; // eax
  __int64 v72; // r11
  __int64 *v73; // rsi
  int v74; // r10d
  __int64 v75; // rdx
  __int64 v76; // rdi
  int v77; // r9d
  int v78; // r10d
  size_t v79; // [rsp+8h] [rbp-2C8h]
  __int64 *v80; // [rsp+10h] [rbp-2C0h]
  _BYTE *v81; // [rsp+18h] [rbp-2B8h]
  _BYTE *v82; // [rsp+20h] [rbp-2B0h]
  __int64 v83; // [rsp+20h] [rbp-2B0h]
  __int64 *v84; // [rsp+20h] [rbp-2B0h]
  __int64 v85; // [rsp+20h] [rbp-2B0h]
  __int64 v86; // [rsp+20h] [rbp-2B0h]
  unsigned __int64 v87; // [rsp+28h] [rbp-2A8h]
  int v88; // [rsp+28h] [rbp-2A8h]
  __int64 v89; // [rsp+28h] [rbp-2A8h]
  unsigned int v90; // [rsp+28h] [rbp-2A8h]
  const char *v91; // [rsp+28h] [rbp-2A8h]
  __int64 v92; // [rsp+28h] [rbp-2A8h]
  __int64 v93; // [rsp+28h] [rbp-2A8h]
  __int64 v94; // [rsp+28h] [rbp-2A8h]
  __int64 *v96; // [rsp+38h] [rbp-298h]
  const char *v97; // [rsp+40h] [rbp-290h] BYREF
  __int64 v98; // [rsp+48h] [rbp-288h]
  _BYTE v99[64]; // [rsp+50h] [rbp-280h] BYREF
  const char **v100; // [rsp+90h] [rbp-240h] BYREF
  __int64 v101; // [rsp+98h] [rbp-238h]
  _QWORD v102[70]; // [rsp+A0h] [rbp-230h] BYREF

  result = (__int64 *)a1[9];
  v80 = (__int64 *)a1[10];
  v96 = result;
  if ( v80 != result )
  {
    while ( 1 )
    {
      v2 = (_QWORD *)*v96;
      v3 = sub_157EBA0(*v96);
      if ( *(_BYTE *)(v3 + 16) != 25 )
        goto LABEL_3;
      v97 = sub_1649960((__int64)v2);
      v98 = v4;
      LOWORD(v102[0]) = 773;
      v100 = &v97;
      v101 = (__int64)".ret";
      v6 = sub_157FBF0(v2, (__int64 *)(v3 + 24), (__int64)&v100);
      v7 = *a1;
      if ( !*a1 )
        goto LABEL_3;
      v8 = *(unsigned int *)(v7 + 48);
      if ( !(_DWORD)v8 )
        goto LABEL_126;
      v9 = *(_QWORD *)(v7 + 32);
      v10 = ((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4);
      v11 = (v8 - 1) & v10;
      v12 = (_QWORD *)(v9 + 16LL * v11);
      v13 = (_QWORD *)*v12;
      if ( v2 != (_QWORD *)*v12 )
      {
        v66 = 1;
        while ( v13 != (_QWORD *)-8LL )
        {
          v67 = v66 + 1;
          v68 = ((_DWORD)v8 - 1) & (v11 + v66);
          v11 = v68;
          v12 = (_QWORD *)(v9 + 16 * v68);
          v13 = (_QWORD *)*v12;
          if ( v2 == (_QWORD *)*v12 )
            goto LABEL_8;
          v66 = v67;
        }
LABEL_126:
        BUG();
      }
LABEL_8:
      if ( v12 == (_QWORD *)(v9 + 16 * v8) )
        goto LABEL_126;
      v14 = v12[1];
      v15 = v99;
      v16 = *(_BYTE **)(v14 + 32);
      v17 = *(_BYTE **)(v14 + 24);
      v97 = v99;
      v18 = 0;
      v98 = 0x800000000LL;
      v19 = v16 - v17;
      v20 = (v16 - v17) >> 3;
      if ( (unsigned __int64)(v16 - v17) > 0x40 )
      {
        v79 = v16 - v17;
        v81 = v16;
        v82 = v17;
        v87 = (v16 - v17) >> 3;
        sub_16CD150((__int64)&v97, v99, v87, 8, v20, v5);
        v18 = v98;
        v19 = v79;
        v16 = v81;
        v17 = v82;
        LODWORD(v20) = v87;
        v15 = (char *)&v97[8 * (unsigned int)v98];
      }
      if ( v17 != v16 )
      {
        v88 = v20;
        memmove(v15, v17, v19);
        v18 = v98;
        LODWORD(v20) = v88;
      }
      LODWORD(v98) = v20 + v18;
      v21 = *a1;
      v22 = *(unsigned int *)(*a1 + 48LL);
      if ( !(_DWORD)v22 )
        goto LABEL_87;
      v23 = *(_QWORD *)(v21 + 32);
      v24 = (v22 - 1) & v10;
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (_QWORD *)*v25;
      if ( v2 != (_QWORD *)*v25 )
        break;
LABEL_15:
      if ( v25 == (_QWORD *)(v23 + 16 * v22) )
        goto LABEL_87;
      v27 = v25[1];
      *(_BYTE *)(v21 + 72) = 0;
      v89 = v21;
      v28 = (__int64 *)sub_22077B0(56);
      v30 = v89;
      v31 = v28;
      if ( !v28 )
        goto LABEL_20;
      *v28 = v6;
      v28[1] = v27;
      if ( v27 )
        v32 = *(_DWORD *)(v27 + 16) + 1;
      else
        v32 = 0;
LABEL_19:
      *((_DWORD *)v31 + 4) = v32;
      v31[3] = 0;
      v31[4] = 0;
      v31[5] = 0;
      v31[6] = -1;
LABEL_20:
      v100 = (const char **)v31;
      v33 = *(_BYTE **)(v27 + 32);
      if ( v33 == *(_BYTE **)(v27 + 40) )
      {
        v92 = v30;
        sub_15CE310(v27 + 24, v33, &v100);
        v30 = v92;
        v35 = *(_DWORD *)(v92 + 48);
        v34 = v92 + 24;
        if ( !v35 )
          goto LABEL_78;
      }
      else
      {
        if ( v33 )
        {
          *(_QWORD *)v33 = v31;
          v33 = *(_BYTE **)(v27 + 32);
        }
        v34 = v30 + 24;
        *(_QWORD *)(v27 + 32) = v33 + 8;
        v35 = *(_DWORD *)(v30 + 48);
        if ( !v35 )
        {
LABEL_78:
          ++*(_QWORD *)(v30 + 24);
          goto LABEL_79;
        }
      }
      v36 = v35 - 1;
      v37 = *(_QWORD *)(v30 + 32);
      v90 = ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4);
      v38 = (v35 - 1) & v90;
      v39 = (__int64 *)(v37 + 16LL * v38);
      v40 = *v39;
      if ( v6 != *v39 )
      {
        v29 = 1;
        v69 = 0;
        while ( v40 != -8 )
        {
          if ( !v69 && v40 == -16 )
            v69 = v39;
          v38 = v36 & (v29 + v38);
          v39 = (__int64 *)(v37 + 16LL * v38);
          v40 = *v39;
          if ( v6 == *v39 )
            goto LABEL_25;
          ++v29;
        }
        v70 = *(_DWORD *)(v30 + 40);
        if ( v69 )
          v39 = v69;
        ++*(_QWORD *)(v30 + 24);
        v62 = v70 + 1;
        if ( 4 * v62 >= 3 * v35 )
        {
LABEL_79:
          v93 = v30;
          sub_15CFCF0(v34, 2 * v35);
          v30 = v93;
          v59 = *(_DWORD *)(v93 + 48);
          if ( !v59 )
            goto LABEL_125;
          v29 = v59 - 1;
          v60 = *(_QWORD *)(v93 + 32);
          LODWORD(v61) = (v59 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v62 = *(_DWORD *)(v93 + 40) + 1;
          v39 = (__int64 *)(v60 + 16LL * (unsigned int)v61);
          v63 = *v39;
          if ( v6 != *v39 )
          {
            v78 = 1;
            v73 = 0;
            while ( v63 != -8 )
            {
              if ( v63 == -16 && !v73 )
                v73 = v39;
              v36 = v78 + 1;
              v61 = v29 & (unsigned int)(v61 + v78);
              v39 = (__int64 *)(v60 + 16 * v61);
              v63 = *v39;
              if ( v6 == *v39 )
                goto LABEL_81;
              ++v78;
            }
LABEL_102:
            if ( v73 )
              v39 = v73;
          }
        }
        else
        {
          v36 = v35 >> 3;
          if ( v35 - *(_DWORD *)(v30 + 44) - v62 <= v35 >> 3 )
          {
            v86 = v30;
            sub_15CFCF0(v34, v35);
            v30 = v86;
            v71 = *(_DWORD *)(v86 + 48);
            if ( !v71 )
            {
LABEL_125:
              ++*(_DWORD *)(v30 + 40);
              BUG();
            }
            v29 = v71 - 1;
            v72 = *(_QWORD *)(v86 + 32);
            v73 = 0;
            v74 = 1;
            LODWORD(v75) = v29 & v90;
            v62 = *(_DWORD *)(v86 + 40) + 1;
            v39 = (__int64 *)(v72 + 16LL * (v29 & v90));
            v76 = *v39;
            if ( v6 != *v39 )
            {
              while ( v76 != -8 )
              {
                if ( !v73 && v76 == -16 )
                  v73 = v39;
                v36 = v74 + 1;
                v75 = v29 & (unsigned int)(v75 + v74);
                v39 = (__int64 *)(v72 + 16 * v75);
                v76 = *v39;
                if ( v6 == *v39 )
                  goto LABEL_81;
                ++v74;
              }
              goto LABEL_102;
            }
          }
        }
LABEL_81:
        *(_DWORD *)(v30 + 40) = v62;
        if ( *v39 != -8 )
          --*(_DWORD *)(v30 + 44);
        *v39 = v6;
        v39[1] = (__int64)v31;
        goto LABEL_29;
      }
LABEL_25:
      v41 = v39[1];
      v39[1] = (__int64)v31;
      if ( v41 )
      {
        v42 = *(_QWORD *)(v41 + 24);
        if ( v42 )
          j_j___libc_free_0(v42, *(_QWORD *)(v41 + 40) - v42);
        j_j___libc_free_0(v41, 56);
        v31 = (__int64 *)v39[1];
      }
LABEL_29:
      v43 = v97;
      v91 = &v97[8 * (unsigned int)v98];
      if ( v97 != v91 )
      {
        while ( 1 )
        {
          v44 = *(_QWORD *)v43;
          *(_BYTE *)(*a1 + 72LL) = 0;
          v45 = *(__int64 **)(v44 + 8);
          if ( v31 != v45 )
            break;
LABEL_56:
          v43 += 8;
          if ( v91 == v43 )
          {
            v91 = v97;
            goto LABEL_58;
          }
        }
        v46 = (char *)v45[4];
        v47 = (char *)v45[3];
        v48 = (v46 - v47) >> 5;
        v49 = (v46 - v47) >> 3;
        if ( v48 > 0 )
        {
          v50 = &v47[32 * v48];
          while ( v44 != *(_QWORD *)v47 )
          {
            if ( v44 == *((_QWORD *)v47 + 1) )
            {
              v47 += 8;
              goto LABEL_38;
            }
            if ( v44 == *((_QWORD *)v47 + 2) )
            {
              v47 += 16;
              goto LABEL_38;
            }
            if ( v44 == *((_QWORD *)v47 + 3) )
            {
              v47 += 24;
              goto LABEL_38;
            }
            v47 += 32;
            if ( v50 == v47 )
            {
              v49 = (v46 - v47) >> 3;
              goto LABEL_61;
            }
          }
          goto LABEL_38;
        }
LABEL_61:
        if ( v49 != 2 )
        {
          if ( v49 != 3 )
          {
            if ( v49 != 1 )
            {
              v47 = (char *)v45[4];
LABEL_38:
              if ( v47 + 8 != v46 )
              {
                v83 = v44;
                memmove(v47, v47 + 8, v46 - (v47 + 8));
                v46 = (char *)v45[4];
                v44 = v83;
              }
              v45[4] = (__int64)(v46 - 8);
              *(_QWORD *)(v44 + 8) = v31;
              v100 = (const char **)v44;
              v51 = (_BYTE *)v31[4];
              if ( v51 == (_BYTE *)v31[5] )
              {
                v85 = v44;
                sub_15CE310((__int64)(v31 + 3), v51, &v100);
                v44 = v85;
              }
              else
              {
                if ( v51 )
                {
                  *(_QWORD *)v51 = v44;
                  v51 = (_BYTE *)v31[4];
                }
                v31[4] = (__int64)(v51 + 8);
              }
              if ( *(_DWORD *)(v44 + 16) != *(_DWORD *)(*(_QWORD *)(v44 + 8) + 16LL) + 1 )
              {
                v102[0] = v44;
                v100 = (const char **)v102;
                v52 = (const char **)v102;
                v84 = v31;
                v101 = 0x4000000001LL;
                LODWORD(v53) = 1;
                do
                {
                  v54 = (unsigned int)v53;
                  v53 = (unsigned int)(v53 - 1);
                  v55 = v52[v54 - 1];
                  LODWORD(v101) = v53;
                  v56 = (const char **)*((_QWORD *)v55 + 4);
                  v57 = (const char **)*((_QWORD *)v55 + 3);
                  *((_DWORD *)v55 + 4) = *(_DWORD *)(*((_QWORD *)v55 + 1) + 16LL) + 1;
                  if ( v57 != v56 )
                  {
                    do
                    {
                      v58 = *v57;
                      if ( *((_DWORD *)*v57 + 4) != *(_DWORD *)(*((_QWORD *)*v57 + 1) + 16LL) + 1 )
                      {
                        if ( HIDWORD(v101) <= (unsigned int)v53 )
                        {
                          sub_16CD150((__int64)&v100, v102, 0, 8, v29, v36);
                          v53 = (unsigned int)v101;
                        }
                        v100[v53] = v58;
                        v53 = (unsigned int)(v101 + 1);
                        LODWORD(v101) = v101 + 1;
                      }
                      ++v57;
                    }
                    while ( v56 != v57 );
                    v52 = v100;
                  }
                }
                while ( (_DWORD)v53 );
                v31 = v84;
                if ( v52 != v102 )
                  _libc_free((unsigned __int64)v52);
              }
              goto LABEL_56;
            }
LABEL_74:
            if ( v44 != *(_QWORD *)v47 )
              v47 = (char *)v45[4];
            goto LABEL_38;
          }
          if ( v44 == *(_QWORD *)v47 )
            goto LABEL_38;
          v47 += 8;
        }
        if ( v44 == *(_QWORD *)v47 )
          goto LABEL_38;
        v47 += 8;
        goto LABEL_74;
      }
LABEL_58:
      if ( v91 != v99 )
        _libc_free((unsigned __int64)v91);
LABEL_3:
      result = ++v96;
      if ( v80 == v96 )
        return result;
    }
    v64 = 1;
    while ( v26 != (_QWORD *)-8LL )
    {
      v77 = v64 + 1;
      v24 = (v22 - 1) & (v64 + v24);
      v25 = (_QWORD *)(v23 + 16LL * v24);
      v26 = (_QWORD *)*v25;
      if ( v2 == (_QWORD *)*v25 )
        goto LABEL_15;
      v64 = v77;
    }
LABEL_87:
    *(_BYTE *)(v21 + 72) = 0;
    v94 = v21;
    v65 = (__int64 *)sub_22077B0(56);
    v30 = v94;
    v31 = v65;
    if ( !v65 )
    {
      v100 = 0;
      BUG();
    }
    *v65 = v6;
    v27 = 0;
    v32 = 0;
    v31[1] = 0;
    goto LABEL_19;
  }
  return result;
}
