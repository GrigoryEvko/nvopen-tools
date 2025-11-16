// Function: sub_293ACB0
// Address: 0x293acb0
//
_BYTE *__fastcall sub_293ACB0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7,
        __int64 a8,
        int a9,
        int a10,
        __int16 a11)
{
  bool v12; // cc
  __int64 **v13; // rdi
  unsigned __int64 v14; // rbx
  unsigned int v15; // eax
  unsigned int k; // ebx
  _DWORD *v17; // r11
  __int64 v18; // r10
  __int64 v19; // rdi
  __int64 (__fastcall *v20)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v21; // rax
  _BYTE *v22; // r13
  int v23; // r13d
  _BYTE *v24; // r14
  int v25; // r12d
  __int64 v26; // rax
  char v27; // al
  __int64 *v28; // rdx
  const char **v29; // rdx
  char v30; // al
  __int64 v31; // rax
  unsigned __int8 *v32; // r12
  __int64 v33; // rdi
  __int64 (__fastcall *v34)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v35; // rax
  _QWORD *v37; // rax
  __int64 v38; // r9
  __int64 v39; // rbx
  __int64 v40; // r14
  __int64 v41; // rdx
  unsigned int v42; // esi
  int v43; // eax
  int v44; // esi
  __int64 v45; // rcx
  char v46; // al
  __int64 *v47; // rdx
  const char **v48; // rdx
  _QWORD *v49; // rax
  __int64 v50; // rsi
  __int64 v51; // r13
  __int64 v52; // r12
  __int64 v53; // rdx
  unsigned int v54; // esi
  _DWORD *v55; // r11
  __int64 v56; // r10
  __int64 v57; // rdi
  __int64 (__fastcall *v58)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64); // rax
  __int64 v59; // rax
  int m; // edx
  __int64 v61; // rsi
  __int64 v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // r14
  __int64 v65; // rax
  __int64 v66; // r13
  __int64 v67; // r13
  __int64 v68; // rbx
  __int64 v69; // rdx
  unsigned int v70; // esi
  _BYTE *v71; // rdi
  int v72; // ecx
  _BYTE *v73; // rcx
  int v74; // eax
  unsigned __int64 v75; // rax
  _DWORD *v76; // rax
  _DWORD *i; // rdx
  __int64 j; // rax
  __int64 v79; // [rsp-10h] [rbp-1C0h]
  __int64 v80; // [rsp+8h] [rbp-1A8h]
  __int64 v81; // [rsp+10h] [rbp-1A0h]
  int v82; // [rsp+2Ch] [rbp-184h]
  __int64 v83; // [rsp+30h] [rbp-180h]
  __int64 v84; // [rsp+38h] [rbp-178h]
  __int64 v85; // [rsp+40h] [rbp-170h]
  void *v86; // [rsp+40h] [rbp-170h]
  unsigned int v87; // [rsp+40h] [rbp-170h]
  __int64 v88; // [rsp+40h] [rbp-170h]
  __int64 v89; // [rsp+40h] [rbp-170h]
  void *v90; // [rsp+40h] [rbp-170h]
  __int64 v91; // [rsp+40h] [rbp-170h]
  _DWORD *v92; // [rsp+50h] [rbp-160h]
  __int64 v93; // [rsp+50h] [rbp-160h]
  _DWORD *v94; // [rsp+50h] [rbp-160h]
  _DWORD *v95; // [rsp+50h] [rbp-160h]
  __int64 v96; // [rsp+50h] [rbp-160h]
  _DWORD *v97; // [rsp+50h] [rbp-160h]
  _BYTE *v100; // [rsp+68h] [rbp-148h]
  _QWORD *v101; // [rsp+68h] [rbp-148h]
  __int64 v102; // [rsp+68h] [rbp-148h]
  unsigned int v103; // [rsp+68h] [rbp-148h]
  const char *v104; // [rsp+70h] [rbp-140h] BYREF
  __int64 v105; // [rsp+78h] [rbp-138h]
  const char *v106; // [rsp+80h] [rbp-130h]
  __int16 v107; // [rsp+90h] [rbp-120h]
  const char **v108; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v109; // [rsp+A8h] [rbp-108h]
  unsigned int v110; // [rsp+B0h] [rbp-100h]
  __int16 v111; // [rsp+C0h] [rbp-F0h]
  _BYTE v112[32]; // [rsp+D0h] [rbp-E0h] BYREF
  __int16 v113; // [rsp+F0h] [rbp-C0h]
  _DWORD *v114; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v115; // [rsp+108h] [rbp-A8h]
  _BYTE s[48]; // [rsp+110h] [rbp-A0h] BYREF
  _DWORD *v117; // [rsp+140h] [rbp-70h] BYREF
  __int64 v118; // [rsp+148h] [rbp-68h]
  _BYTE v119[96]; // [rsp+150h] [rbp-60h] BYREF

  v12 = *(_DWORD *)(a3 + 8) <= 1u;
  v13 = *(__int64 ***)a3;
  v14 = *(unsigned int *)(*(_QWORD *)a3 + 32LL);
  v114 = s;
  v115 = 0xC00000000LL;
  v82 = v14;
  v117 = v119;
  v118 = 0xC00000000LL;
  if ( !v12 )
  {
    if ( v14 )
    {
      v71 = s;
      if ( v14 > 0xC )
      {
        sub_C8D5F0((__int64)&v114, s, v14, 4u, (__int64)&v114, a6);
        v71 = &v114[(unsigned int)v115];
      }
      memset(v71, 255, 4 * v14);
      v72 = *(_DWORD *)(a3 + 8);
      LODWORD(v115) = v14 + v115;
      if ( !v72 )
      {
        v75 = (unsigned int)v118;
        if ( v14 == (unsigned int)v118 )
          goto LABEL_95;
LABEL_86:
        if ( v14 >= v75 )
        {
          if ( v14 > HIDWORD(v118) )
          {
            sub_C8D5F0((__int64)&v117, v119, v14, 4u, a5, a6);
            v75 = (unsigned int)v118;
          }
          v76 = &v117[v75];
          for ( i = &v117[v14]; i != v76; ++v76 )
          {
            if ( v76 )
              *v76 = 0;
          }
        }
        LODWORD(v118) = v14;
LABEL_94:
        if ( !(_DWORD)v14 )
        {
LABEL_97:
          v13 = *(__int64 ***)a3;
          goto LABEL_2;
        }
LABEL_95:
        for ( j = 0; j != v14; ++j )
          v117[j] = j;
        goto LABEL_97;
      }
      v73 = v114;
    }
    else
    {
      v73 = s;
    }
    v74 = 0;
    while ( 1 )
    {
      *(_DWORD *)&v73[4 * v74] = v74;
      if ( *(_DWORD *)(a3 + 8) <= (unsigned int)++v74 )
        break;
      v73 = v114;
    }
    v75 = (unsigned int)v118;
    if ( v14 == (unsigned int)v118 )
      goto LABEL_94;
    goto LABEL_86;
  }
LABEL_2:
  v100 = (_BYTE *)sub_ACADE0(v13);
  v15 = *(_DWORD *)(a3 + 12);
  if ( v15 )
  {
    for ( k = 0; v15 > k; ++k )
    {
      while ( 1 )
      {
        v23 = *(_DWORD *)(a3 + 8);
        v24 = *(_BYTE **)(a2 + 8LL * k);
        v25 = v23;
        if ( v15 - 1 == k )
        {
          v26 = *(_QWORD *)(a3 + 24);
          if ( v26 )
            break;
        }
        if ( v23 == 1 )
          goto LABEL_16;
LABEL_5:
        v17 = v114;
        v111 = 257;
        v18 = (unsigned int)v115;
        v19 = a1[10];
        v20 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v19 + 112LL);
        if ( v20 == sub_9B6630 )
        {
          if ( *v24 > 0x15u )
            goto LABEL_34;
          v92 = v114;
          v85 = (unsigned int)v115;
          v21 = sub_AD5CE0((__int64)v24, (__int64)v24, v114, (unsigned int)v115, 0);
          v18 = v85;
          v17 = v92;
          v22 = (_BYTE *)v21;
        }
        else
        {
          v95 = v114;
          v89 = (unsigned int)v115;
          v62 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v20)(
                  v19,
                  v24,
                  v24,
                  v114,
                  (unsigned int)v115);
          v17 = v95;
          v18 = v89;
          v22 = (_BYTE *)v62;
        }
        if ( v22 )
          goto LABEL_9;
LABEL_34:
        v86 = v17;
        v93 = v18;
        v113 = 257;
        v37 = sub_BD2C40(112, unk_3F1FE60);
        v22 = v37;
        if ( v37 )
        {
          sub_B4E9E0((__int64)v37, (__int64)v24, (__int64)v24, v86, v93, (__int64)v112, 0, 0);
          v38 = v79;
        }
        (*(void (__fastcall **)(__int64, _BYTE *, const char ***, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
          a1[11],
          v22,
          &v108,
          a1[7],
          a1[8],
          v38);
        if ( *a1 == *a1 + 16LL * *((unsigned int *)a1 + 2) )
        {
LABEL_9:
          if ( k )
            goto LABEL_40;
          goto LABEL_10;
        }
        v87 = k;
        v39 = *a1;
        v40 = *a1 + 16LL * *((unsigned int *)a1 + 2);
        do
        {
          v41 = *(_QWORD *)(v39 + 8);
          v42 = *(_DWORD *)v39;
          v39 += 16;
          sub_B99FD0((__int64)v22, v42, v41);
        }
        while ( v40 != v39 );
        k = v87;
        if ( v87 )
        {
LABEL_40:
          v43 = 0;
          if ( v25 )
          {
            do
            {
              v44 = v82 + v43;
              v45 = k * *(_DWORD *)(a3 + 8) + v43++;
              v117[v45] = v44;
            }
            while ( v25 != v43 );
          }
          v46 = a11;
          if ( (_BYTE)a11 )
          {
            if ( (_BYTE)a11 == 1 )
            {
              v104 = ".upto";
              v46 = 3;
              v107 = 259;
            }
            else
            {
              if ( HIBYTE(a11) == 1 )
              {
                v47 = a7;
                v81 = a8;
              }
              else
              {
                v47 = (__int64 *)&a7;
                v46 = 2;
              }
              v104 = (const char *)v47;
              v106 = ".upto";
              v105 = v81;
              LOBYTE(v107) = v46;
              HIBYTE(v107) = 3;
            }
            if ( HIBYTE(v107) == 1 )
            {
              v48 = (const char **)v104;
              v80 = v105;
            }
            else
            {
              v48 = &v104;
              v46 = 2;
            }
            v108 = v48;
            v110 = k;
            v109 = v80;
            LOBYTE(v111) = v46;
            HIBYTE(v111) = 9;
          }
          else
          {
            v107 = 256;
            v111 = 256;
          }
          v55 = v117;
          v56 = (unsigned int)v118;
          v57 = a1[10];
          v58 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v57 + 112LL);
          if ( v58 == sub_9B6630 )
          {
            if ( *v100 <= 0x15u && *v22 <= 0x15u )
            {
              v94 = v117;
              v88 = (unsigned int)v118;
              v59 = sub_AD5CE0((__int64)v100, (__int64)v22, v117, (unsigned int)v118, 0);
              v56 = v88;
              v55 = v94;
              goto LABEL_60;
            }
            goto LABEL_70;
          }
          v97 = v117;
          v91 = (unsigned int)v118;
          v59 = ((__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, _DWORD *, _QWORD))v58)(
                  v57,
                  v100,
                  v22,
                  v117,
                  (unsigned int)v118);
          v55 = v97;
          v56 = v91;
LABEL_60:
          if ( v59 )
          {
            v100 = (_BYTE *)v59;
          }
          else
          {
LABEL_70:
            v96 = v56;
            v90 = v55;
            v113 = 257;
            v63 = sub_BD2C40(112, unk_3F1FE60);
            v64 = (__int64)v63;
            if ( v63 )
              sub_B4E9E0((__int64)v63, (__int64)v100, (__int64)v22, v90, v96, (__int64)v112, 0, 0);
            (*(void (__fastcall **)(__int64, __int64, const char ***, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
              a1[11],
              v64,
              &v108,
              a1[7],
              a1[8]);
            v65 = *a1;
            v66 = 16LL * *((unsigned int *)a1 + 2);
            if ( v65 != v65 + v66 )
            {
              v103 = k;
              v67 = v65 + v66;
              v68 = *a1;
              do
              {
                v69 = *(_QWORD *)(v68 + 8);
                v70 = *(_DWORD *)v68;
                v68 += 16;
                sub_B99FD0(v64, v70, v69);
              }
              while ( v67 != v68 );
              k = v103;
            }
            v100 = (_BYTE *)v64;
          }
          if ( v25 )
          {
            for ( m = 0; m != v25; ++m )
            {
              v61 = k * *(_DWORD *)(a3 + 8) + m;
              v117[v61] = v61;
            }
          }
          goto LABEL_11;
        }
LABEL_10:
        v100 = v22;
LABEL_11:
        v15 = *(_DWORD *)(a3 + 12);
        if ( v15 <= ++k )
          goto LABEL_29;
      }
      if ( *(_BYTE *)(v26 + 8) == 17 )
      {
        v25 = *(_DWORD *)(v26 + 32);
        if ( v25 != 1 )
          goto LABEL_5;
      }
LABEL_16:
      v27 = a11;
      if ( (_BYTE)a11 )
      {
        if ( (_BYTE)a11 == 1 )
        {
          v29 = (const char **)".upto";
          v104 = ".upto";
          v83 = v105;
          v107 = 259;
          v30 = 3;
        }
        else
        {
          if ( HIBYTE(a11) == 1 )
          {
            v28 = a7;
            v84 = a8;
          }
          else
          {
            v28 = (__int64 *)&a7;
            v27 = 2;
          }
          v104 = (const char *)v28;
          v29 = &v104;
          LOBYTE(v107) = v27;
          v30 = 2;
          v105 = v84;
          v106 = ".upto";
          HIBYTE(v107) = 3;
        }
        v108 = v29;
        v110 = k;
        v109 = v83;
        LOBYTE(v111) = v30;
        HIBYTE(v111) = 9;
      }
      else
      {
        v107 = 256;
        v111 = 256;
      }
      v31 = sub_BCB2E0((_QWORD *)a1[9]);
      v32 = (unsigned __int8 *)sub_ACD640(v31, k * v23, 0);
      v33 = a1[10];
      v34 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v33 + 104LL);
      if ( v34 == sub_948040 )
      {
        if ( *v100 > 0x15u || *v24 > 0x15u || *v32 > 0x15u )
        {
LABEL_50:
          v113 = 257;
          v49 = sub_BD2C40(72, 3u);
          if ( v49 )
          {
            v50 = (__int64)v100;
            v101 = v49;
            sub_B4DFA0((__int64)v49, v50, (__int64)v24, (__int64)v32, (__int64)v112, 0, 0, 0);
            v49 = v101;
          }
          v102 = (__int64)v49;
          (*(void (__fastcall **)(__int64, _QWORD *, const char ***, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
            a1[11],
            v49,
            &v108,
            a1[7],
            a1[8]);
          v51 = *a1;
          v35 = v102;
          v52 = *a1 + 16LL * *((unsigned int *)a1 + 2);
          if ( *a1 != v52 )
          {
            do
            {
              v53 = *(_QWORD *)(v51 + 8);
              v54 = *(_DWORD *)v51;
              v51 += 16;
              sub_B99FD0(v102, v54, v53);
            }
            while ( v52 != v51 );
            v35 = v102;
          }
          goto LABEL_28;
        }
        v35 = sub_AD5A90((__int64)v100, v24, v32, 0);
      }
      else
      {
        v35 = v34(v33, v100, v24, v32);
      }
      if ( !v35 )
        goto LABEL_50;
LABEL_28:
      v100 = (_BYTE *)v35;
      v15 = *(_DWORD *)(a3 + 12);
    }
  }
LABEL_29:
  if ( v117 != (_DWORD *)v119 )
    _libc_free((unsigned __int64)v117);
  if ( v114 != (_DWORD *)s )
    _libc_free((unsigned __int64)v114);
  return v100;
}
