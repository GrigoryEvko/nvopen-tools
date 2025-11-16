// Function: sub_35BFF10
// Address: 0x35bff10
//
__int64 __fastcall sub_35BFF10(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rbx
  _QWORD *v3; // rax
  __int64 v4; // rdi
  __int64 result; // rax
  __int64 v6; // r13
  unsigned int v7; // r12d
  int v8; // ebx
  __int64 v9; // rax
  double v10; // xmm1_8
  __int64 v11; // rax
  float v12; // xmm1_4
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edi
  unsigned int v16; // ecx
  int *v17; // rsi
  __int64 v18; // r8
  int *v19; // r9
  __int64 v20; // rax
  unsigned int v21; // ecx
  int *v22; // rsi
  int v23; // r10d
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rcx
  _QWORD *v27; // rax
  __int64 v28; // r12
  unsigned int *v29; // rdx
  unsigned int *v30; // rcx
  __int64 v31; // rbx
  __int64 v32; // r9
  __int64 v33; // rax
  _DWORD *v34; // rax
  unsigned int v35; // r8d
  unsigned __int64 v36; // r15
  char *v37; // rax
  unsigned int v38; // r8d
  unsigned int v39; // r9d
  char *v40; // rdi
  char *v41; // rax
  char *v42; // rax
  unsigned int v43; // r15d
  unsigned int v44; // r10d
  __int64 v45; // rdx
  int v46; // esi
  unsigned int v47; // eax
  __int64 v48; // rcx
  float *v49; // rdx
  unsigned __int64 v50; // rdi
  _QWORD *v51; // r15
  __int64 v52; // rax
  __int64 v53; // rsi
  __int64 v54; // r15
  unsigned int v55; // edx
  int *v56; // rcx
  int v57; // edi
  unsigned int **v58; // rsi
  unsigned int *v59; // rax
  unsigned int v60; // edx
  _DWORD *v61; // rax
  unsigned int v62; // ebx
  __int64 v63; // rax
  __int64 v64; // rsi
  size_t v65; // r12
  void *v66; // rdi
  _QWORD *v67; // r15
  volatile signed __int32 *v68; // rdi
  volatile signed __int32 *v69; // rbx
  __int64 v70; // rdx
  char *v71; // rax
  size_t v72; // rdx
  int v73; // r8d
  char *v74; // rdi
  char *v75; // rax
  char *v76; // rax
  unsigned int v77; // r11d
  unsigned int v78; // r9d
  __int64 v79; // rdx
  int v80; // esi
  unsigned int v81; // eax
  __int64 v82; // rcx
  float *v83; // rdx
  int v84; // esi
  int v85; // r11d
  int v86; // esi
  int v87; // ecx
  int v88; // r9d
  int v89; // r8d
  __int64 v91; // [rsp+10h] [rbp-E0h]
  __int64 v92; // [rsp+18h] [rbp-D8h]
  const void **v93; // [rsp+20h] [rbp-D0h]
  int v94; // [rsp+20h] [rbp-D0h]
  unsigned int v95; // [rsp+2Ch] [rbp-C4h]
  unsigned int v96; // [rsp+30h] [rbp-C0h]
  __int64 v97; // [rsp+30h] [rbp-C0h]
  int v98; // [rsp+30h] [rbp-C0h]
  unsigned int n; // [rsp+38h] [rbp-B8h]
  unsigned int na; // [rsp+38h] [rbp-B8h]
  __int64 v101; // [rsp+40h] [rbp-B0h]
  int v102; // [rsp+40h] [rbp-B0h]
  __int64 v103; // [rsp+48h] [rbp-A8h]
  unsigned int v104; // [rsp+50h] [rbp-A0h]
  __int64 v105; // [rsp+50h] [rbp-A0h]
  int v106; // [rsp+50h] [rbp-A0h]
  __int64 *v107; // [rsp+58h] [rbp-98h]
  __int64 v108; // [rsp+70h] [rbp-80h] BYREF
  volatile signed __int32 *v109; // [rsp+78h] [rbp-78h]
  unsigned __int64 v110; // [rsp+80h] [rbp-70h] BYREF
  unsigned __int64 v111; // [rsp+88h] [rbp-68h]
  _QWORD *v112; // [rsp+90h] [rbp-60h] BYREF
  __int64 v113; // [rsp+98h] [rbp-58h]
  __int64 v114; // [rsp+A0h] [rbp-50h]
  __int16 v115; // [rsp+A8h] [rbp-48h]
  char v116; // [rsp+AAh] [rbp-46h]
  __int64 v117; // [rsp+B0h] [rbp-40h]

  v107 = (__int64 *)a2[2];
  v2 = *a2;
  v92 = *a2;
  v3 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a2 + 16LL) + 200LL))(*(_QWORD *)(*a2 + 16LL));
  v4 = *(_QWORD *)(v2 + 328);
  v113 = 0;
  v112 = v3;
  result = v2 + 320;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v103 = v4;
  v91 = v2 + 320;
  if ( v4 != v2 + 320 )
  {
    while ( 1 )
    {
      v6 = *(_QWORD *)(v103 + 56);
      if ( v6 != v103 + 48 )
        break;
LABEL_38:
      result = *(_QWORD *)(v103 + 8);
      v103 = result;
      if ( v91 == result )
        return result;
    }
    while ( 1 )
    {
      if ( !(unsigned __int8)sub_2F66060(&v112, v6) )
        goto LABEL_35;
      v7 = v113;
      v8 = HIDWORD(v113);
      if ( (_DWORD)v113 == HIDWORD(v113) )
        goto LABEL_35;
      v9 = sub_2E39EA0(v107, v103);
      v10 = v9 < 0
          ? (double)(int)(v9 & 1 | ((unsigned __int64)v9 >> 1)) + (double)(int)(v9 & 1 | ((unsigned __int64)v9 >> 1))
          : (double)(int)v9;
      v11 = sub_2E3A080((__int64)v107);
      if ( v11 < 0 )
      {
        v12 = v10
            / ((double)(int)(v11 & 1 | ((unsigned __int64)v11 >> 1))
             + (double)(int)(v11 & 1 | ((unsigned __int64)v11 >> 1)));
        if ( !v117 )
        {
LABEL_44:
          v51 = *(_QWORD **)(v92 + 32);
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v51 + 16LL) + 200LL))(*(_QWORD *)(*v51 + 16LL))
                                                + 248)
                                    + 16LL)
                        + v7)
            && (*(_QWORD *)(v51[48] + 8LL * (v7 >> 6)) & (1LL << v7)) == 0 )
          {
            v52 = *((unsigned int *)a2 + 12);
            v53 = a2[4];
            v54 = 0x5FFFFFFFA0LL;
            if ( !(_DWORD)v52 )
              goto LABEL_50;
            v55 = (v52 - 1) & (37 * v8);
            v56 = (int *)(v53 + 8LL * v55);
            v57 = *v56;
            if ( v8 == *v56 )
            {
LABEL_48:
              if ( v56 != (int *)(v53 + 8 * v52) )
              {
                v54 = 96LL * (unsigned int)v56[1];
                goto LABEL_50;
              }
            }
            else
            {
              v87 = 1;
              while ( v57 != -1 )
              {
                v89 = v87 + 1;
                v55 = (v52 - 1) & (v87 + v55);
                v56 = (int *)(v53 + 8LL * v55);
                v57 = *v56;
                if ( v8 == *v56 )
                  goto LABEL_48;
                v87 = v89;
              }
            }
            v54 = 0x5FFFFFFFA0LL;
LABEL_50:
            v58 = (unsigned int **)(v54 + a2[20]);
            v59 = v58[6];
            v60 = *v59;
            if ( *v59 )
            {
              v61 = (_DWORD *)*((_QWORD *)v59 + 1);
              v62 = 0;
              while ( 1 )
              {
                ++v62;
                if ( v7 == *v61 )
                  break;
                ++v61;
                if ( v60 == v62 )
                  goto LABEL_35;
              }
              v63 = (__int64)*v58;
              v64 = **v58;
              v105 = v63;
              v65 = 4 * v64;
              v66 = (void *)sub_2207820(4 * v64);
              if ( v66 && v64 )
                v66 = memset(v66, 0, v65);
              if ( v65 )
                v66 = memmove(v66, *(const void **)(v105 + 8), v65);
              v111 = (unsigned __int64)v66;
              *((float *)v66 + v62) = *((float *)v66 + v62) - v12;
              LODWORD(v110) = v64;
              sub_35BE780(&v108, (__int64)(a2 + 11), (unsigned int *)&v110);
              if ( v111 )
                j_j___libc_free_0_0(v111);
              v67 = (_QWORD *)(a2[20] + v54);
              v68 = (volatile signed __int32 *)v67[1];
              *v67 = v108;
              v69 = v109;
              if ( v109 != v68 )
              {
                if ( v109 )
                {
                  if ( &_pthread_key_create )
                    _InterlockedAdd(v109 + 2, 1u);
                  else
                    ++*((_DWORD *)v109 + 2);
                  v68 = (volatile signed __int32 *)v67[1];
                }
                if ( v68 )
                  sub_A191D0(v68);
                v67[1] = v69;
                v68 = v109;
              }
              if ( v68 )
                sub_A191D0(v68);
            }
            goto LABEL_35;
          }
          goto LABEL_35;
        }
      }
      else
      {
        v12 = v10 / (double)(int)v11;
        if ( !v117 )
          goto LABEL_44;
      }
      v13 = *((unsigned int *)a2 + 12);
      v14 = a2[4];
      if ( !(_DWORD)v13 )
        break;
      v15 = v13 - 1;
      v16 = (v13 - 1) & (37 * v7);
      v17 = (int *)(v14 + 8LL * v16);
      LODWORD(v18) = *v17;
      if ( v7 == *v17 )
      {
LABEL_11:
        v19 = (int *)(v14 + 8 * v13);
        if ( v17 == v19 )
        {
          v20 = 0x5FFFFFFFA0LL;
          LODWORD(v18) = -1;
        }
        else
        {
          v18 = (unsigned int)v17[1];
          v20 = 96 * v18;
        }
      }
      else
      {
        v86 = 1;
        while ( (_DWORD)v18 != -1 )
        {
          v88 = v86 + 1;
          v16 = v15 & (v86 + v16);
          v17 = (int *)(v14 + 8LL * v16);
          LODWORD(v18) = *v17;
          if ( v7 == *v17 )
            goto LABEL_11;
          v86 = v88;
        }
        v19 = (int *)(v14 + 8 * v13);
        v20 = 0x5FFFFFFFA0LL;
      }
      v21 = v15 & (37 * v8);
      v22 = (int *)(v14 + 8LL * v21);
      v23 = *v22;
      if ( v8 != *v22 )
      {
        v84 = 1;
        while ( v23 != -1 )
        {
          v85 = v84 + 1;
          v21 = v15 & (v84 + v21);
          v22 = (int *)(v14 + 8LL * v21);
          v23 = *v22;
          if ( v8 == *v22 )
            goto LABEL_14;
          v84 = v85;
        }
        goto LABEL_75;
      }
LABEL_14:
      if ( v22 == v19 )
        goto LABEL_75;
      v24 = (unsigned int)v22[1];
      v25 = 96 * v24;
LABEL_16:
      v26 = a2[20];
      v27 = (_QWORD *)(v26 + v20);
      v28 = *(_QWORD *)(v26 + v25 + 48);
      v29 = (unsigned int *)v27[9];
      v30 = (unsigned int *)v27[10];
      v31 = v27[6];
      if ( v29 == v30 )
      {
LABEL_76:
        v94 = v18;
        v102 = *(_DWORD *)v31 + 1;
        v106 = *(_DWORD *)v28 + 1;
        v97 = (unsigned int)(v102 * v106);
        v71 = (char *)sub_2207820(4 * v97);
        v72 = 4 * v97;
        v73 = v94;
        v74 = v71;
        if ( v71 && v102 * v106 )
        {
          v75 = (char *)memset(v71, 0, v72);
          v72 = 4 * v97;
          v73 = v94;
          v74 = v75;
        }
        if ( v72 )
        {
          v98 = v73;
          v76 = (char *)memset(v74, 0, v72);
          v73 = v98;
          v74 = v76;
        }
        v77 = v106;
        v78 = 0;
        if ( *(_DWORD *)v31 )
        {
          do
          {
            v79 = v78++;
            v80 = *(_DWORD *)(*(_QWORD *)(v31 + 8) + 4 * v79);
            v81 = 0;
            if ( *(_DWORD *)v28 )
            {
              do
              {
                v82 = v81++;
                if ( v80 == *(_DWORD *)(*(_QWORD *)(v28 + 8) + 4 * v82) )
                {
                  v83 = (float *)&v74[4 * v77 + 4 * (unsigned __int64)v81];
                  *v83 = *v83 - v12;
                }
              }
              while ( v81 != *(_DWORD *)v28 );
            }
            v77 += v106;
          }
          while ( v78 != *(_DWORD *)v31 );
        }
        v111 = (unsigned __int64)v74;
        v110 = __PAIR64__(v106, v102);
        sub_35BFD50(a2, v73, v24, (__int64 *)&v110);
        v50 = v111;
        if ( !v111 )
          goto LABEL_35;
      }
      else
      {
        while ( 1 )
        {
          v32 = *v29;
          v33 = a2[26] + 48 * v32;
          if ( *(_DWORD *)(v33 + 20) == (_DWORD)v24 )
          {
            if ( (_DWORD)v32 == -1 )
              goto LABEL_76;
            goto LABEL_22;
          }
          if ( *(_DWORD *)(v33 + 24) == (_DWORD)v24 )
            break;
          if ( v30 == ++v29 )
            goto LABEL_76;
        }
        if ( (_DWORD)v32 == -1 )
          goto LABEL_76;
        v70 = v28;
        v28 = v31;
        v31 = v70;
LABEL_22:
        v34 = *(_DWORD **)v33;
        v95 = v32;
        v35 = v34[1];
        v93 = (const void **)v34;
        v104 = *v34;
        n = v35;
        v36 = 4LL * v35 * *v34;
        v101 = v35 * *v34;
        v37 = (char *)sub_2207820(v36);
        v38 = n;
        v39 = v95;
        v40 = v37;
        if ( v37 && v101 )
        {
          v41 = (char *)memset(v37, 0, v36);
          v39 = v95;
          v38 = n;
          v40 = v41;
        }
        if ( v36 )
        {
          v96 = v38;
          na = v39;
          v42 = (char *)memmove(v40, v93[1], v36);
          v38 = v96;
          v39 = na;
          v40 = v42;
        }
        v43 = v38;
        v44 = 0;
        if ( *(_DWORD *)v28 )
        {
          do
          {
            v45 = v44++;
            v46 = *(_DWORD *)(*(_QWORD *)(v28 + 8) + 4 * v45);
            v47 = 0;
            if ( *(_DWORD *)v31 )
            {
              do
              {
                v48 = v47++;
                if ( v46 == *(_DWORD *)(*(_QWORD *)(v31 + 8) + 4 * v48) )
                {
                  v49 = (float *)&v40[4 * v43 + 4 * (unsigned __int64)v47];
                  *v49 = *v49 - v12;
                }
              }
              while ( v47 != *(_DWORD *)v31 );
            }
            v43 += v38;
          }
          while ( v44 != *(_DWORD *)v28 );
        }
        v111 = (unsigned __int64)v40;
        v110 = __PAIR64__(v38, v104);
        sub_35BFB40((__int64)a2, v39, (__int64 *)&v110);
        v50 = v111;
        if ( !v111 )
          goto LABEL_35;
      }
      j_j___libc_free_0_0(v50);
LABEL_35:
      if ( !v6 )
        BUG();
      if ( (*(_BYTE *)v6 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v6 + 44) & 8) != 0 )
          v6 = *(_QWORD *)(v6 + 8);
      }
      v6 = *(_QWORD *)(v6 + 8);
      if ( v103 + 48 == v6 )
        goto LABEL_38;
    }
    v20 = 0x5FFFFFFFA0LL;
    LODWORD(v18) = -1;
LABEL_75:
    v25 = 0x5FFFFFFFA0LL;
    LODWORD(v24) = -1;
    goto LABEL_16;
  }
  return result;
}
