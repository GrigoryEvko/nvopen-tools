// Function: sub_16904A0
// Address: 0x16904a0
//
__int64 __fastcall sub_16904A0(size_t *a1, const char **a2, unsigned int *a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  size_t v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // edx
  size_t v25; // rax
  const char *v26; // r15
  __int64 v27; // r14
  size_t v28; // rax
  __int64 v29; // r14
  size_t v30; // rax
  __int64 (__fastcall *v31)(_QWORD, _QWORD); // r15
  const char *v32; // rax
  __int64 v33; // rax
  unsigned int v34; // r9d
  size_t v35; // rsi
  __int64 v36; // r12
  size_t v37; // r13
  __int64 v38; // rax
  __int64 v39; // r14
  unsigned int v41; // r13d
  __int64 v42; // r13
  __int64 v43; // r12
  const char *v44; // rax
  unsigned int v45; // r12d
  __int64 v46; // rax
  const char *v47; // rbx
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned int v51; // r12d
  __int64 v52; // r13
  const char *v53; // rax
  unsigned int v54; // r13d
  __int64 v55; // r13
  __int64 v56; // rsi
  const char *v57; // rax
  unsigned int v58; // r13d
  __int64 v59; // rax
  unsigned int v60; // r13d
  __int64 (__fastcall *v61)(const char **); // rax
  __int64 v62; // r13
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdx
  const char *v66; // rax
  __int64 v67; // r13
  __int64 v68; // rax
  unsigned int v69; // r13d
  __int64 (__fastcall *v70)(const char **); // rax
  __int64 v71; // r13
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned int v74; // r12d
  char *v75; // r13
  __int64 v76; // rax
  char *v77; // rbx
  char v78; // r12
  char *v79; // rax
  void *v80; // r15
  __int64 v81; // rax
  const char *v82; // rax
  unsigned int v83; // r14d
  __int64 v84; // rax
  __int64 v85; // rdx
  __int64 v86; // r8
  __int64 v87; // r9
  int v88; // eax
  int v89; // r15d
  __int64 v90; // rcx
  size_t v91; // rax
  __int64 v92; // rax
  __int64 v93; // [rsp-10h] [rbp-1C0h]
  __int64 v94; // [rsp-8h] [rbp-1B8h]
  int v95; // [rsp+Ch] [rbp-1A4h]
  int v96; // [rsp+10h] [rbp-1A0h]
  size_t v97; // [rsp+18h] [rbp-198h]
  __int64 v98; // [rsp+18h] [rbp-198h]
  __int64 v100; // [rsp+20h] [rbp-190h]
  size_t n; // [rsp+28h] [rbp-188h]
  unsigned int na; // [rsp+28h] [rbp-188h]
  _QWORD v103[2]; // [rsp+30h] [rbp-180h] BYREF
  _QWORD v104[2]; // [rsp+40h] [rbp-170h] BYREF
  _QWORD v105[2]; // [rsp+50h] [rbp-160h] BYREF
  __int16 v106; // [rsp+60h] [rbp-150h]
  _BYTE *v107; // [rsp+70h] [rbp-140h] BYREF
  __int64 v108; // [rsp+78h] [rbp-138h]
  _BYTE v109[304]; // [rsp+80h] [rbp-130h] BYREF

  v6 = sub_1691920(a1[1], *(unsigned __int16 *)(*a1 + 42));
  n = v6;
  if ( v6 )
  {
    LODWORD(v97) = v7;
    v8 = sub_1691920(v7, *(unsigned __int16 *)(v6 + 42));
    v10 = v8;
    if ( v8 )
    {
      LODWORD(v97) = v9;
      v11 = sub_1691920(v9, *(unsigned __int16 *)(v8 + 42));
      n = v11;
      if ( !v11 )
        goto LABEL_24;
      LODWORD(v97) = v12;
      v13 = sub_1691920(v12, *(unsigned __int16 *)(v11 + 42));
      v10 = v13;
      if ( !v13 )
        goto LABEL_11;
      LODWORD(v97) = v14;
      v15 = sub_1691920(v14, *(unsigned __int16 *)(v13 + 42));
      n = v15;
      if ( !v15 )
      {
LABEL_24:
        n = v10;
        goto LABEL_11;
      }
      LODWORD(v97) = v16;
      v17 = sub_1691920(v16, *(unsigned __int16 *)(v15 + 42));
      v10 = v17;
      if ( v17 )
      {
        LODWORD(v97) = v18;
        v19 = sub_1691920(v18, *(unsigned __int16 *)(v17 + 42));
        n = v19;
        if ( v19 )
        {
          LODWORD(v97) = v20;
          v21 = sub_1691920(v20, *(unsigned __int16 *)(v19 + 42));
          if ( v21 )
          {
            LODWORD(v97) = v22;
            n = v21;
            v107 = (_BYTE *)sub_1691920(v22, *(unsigned __int16 *)(v21 + 42));
            v108 = v23;
            if ( v107 )
            {
              n = sub_168F990(&v107);
              LODWORD(v97) = v24;
            }
          }
          goto LABEL_11;
        }
        goto LABEL_24;
      }
    }
LABEL_11:
    v25 = *a1;
    goto LABEL_12;
  }
  v25 = *a1;
  n = *a1;
  v97 = a1[1];
LABEL_12:
  v26 = *a2;
  if ( *(_DWORD *)(v25 + 32) == *(_DWORD *)(n + 32) )
  {
    v96 = a4;
    LODWORD(v26) = (*(__int64 (__fastcall **)(const char **, _QWORD))v26)(a2, *a3);
  }
  else
  {
    v27 = *(_QWORD *)(n + 8);
    v28 = 0;
    if ( v27 )
      v28 = strlen(*(const char **)(n + 8));
    v104[1] = v28;
    v104[0] = v27;
    v29 = **(_QWORD **)n;
    v30 = 0;
    if ( v29 )
      v30 = strlen(**(const char ***)n);
    v103[1] = v30;
    v105[0] = v103;
    v103[0] = v29;
    v105[1] = v104;
    v106 = 1285;
    v108 = 0x10000000000LL;
    v107 = v109;
    v31 = (__int64 (__fastcall *)(_QWORD, _QWORD))*((_QWORD *)v26 + 2);
    sub_16E2F40(v105, &v107);
    v26 = (const char *)((__int64 (__fastcall *)(const char **, _BYTE *, _QWORD))v31)(a2, v107, (unsigned int)v108);
    if ( v107 != v109 )
      _libc_free((unsigned __int64)v107);
    v96 = 0;
    if ( v26 )
      v96 = strlen(v26);
  }
  switch ( *(_BYTE *)(*a1 + 36) )
  {
    case 3:
      v44 = (const char *)(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      if ( a4 != strlen(v44) )
        return 0;
      v45 = (*a3)++;
      v46 = sub_22077B0(80);
      v39 = v46;
      if ( v46 )
        sub_16933F0(v46, n, v97, (_DWORD)v26, v96, v45, 0);
      v47 = *(const char **)(*a1 + 48);
      if ( !v47 )
      {
        if ( *(_BYTE *)(n + 36) != 4 )
          return v39;
        goto LABEL_44;
      }
      if ( *v47 )
      {
        v48 = *(unsigned int *)(v39 + 56);
        do
        {
          if ( *(_DWORD *)(v39 + 60) <= (unsigned int)v48 )
          {
            sub_16CD150(v39 + 48, v39 + 64, 0, 8);
            v48 = *(unsigned int *)(v39 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v48) = v47;
          v48 = (unsigned int)(*(_DWORD *)(v39 + 56) + 1);
          *(_DWORD *)(v39 + 56) = v48;
          v47 += strlen(v47) + 1;
        }
        while ( *v47 );
        if ( *(_BYTE *)(n + 36) == 4 && !*(_QWORD *)(*a1 + 48) )
        {
LABEL_44:
          v49 = *(unsigned int *)(v39 + 56);
          if ( (unsigned int)v49 >= *(_DWORD *)(v39 + 60) )
          {
            sub_16CD150(v39 + 48, v39 + 64, 0, 8);
            v49 = *(unsigned int *)(v39 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v49) = byte_3F871B3;
          ++*(_DWORD *)(v39 + 56);
        }
      }
      return v39;
    case 4:
      v50 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      v51 = *a3;
      v52 = v50 + a4;
      ++*a3;
      v39 = sub_22077B0(80);
      if ( v39 )
        sub_1693430(v39, n, v97, (_DWORD)v26, v96, v51, v52, 0);
      return v39;
    case 5:
    case 0xC:
      v41 = *a3 + 2;
      *a3 = v41;
      if ( v41 > (*((unsigned int (__fastcall **)(const char **))*a2 + 1))(a2)
        || !(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3 - 1) )
      {
        return 0;
      }
      v42 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3 - 2) + a4;
      v43 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3 - 1);
      v39 = sub_22077B0(80);
      if ( v39 )
        sub_1693480(v39, n, v97, (_DWORD)v26, v96, *a3 - 2, v42, v43, 0);
      return v39;
    case 6:
      v53 = (const char *)(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      if ( a4 != strlen(v53) )
        return 0;
      goto LABEL_50;
    case 7:
      v56 = *a3;
      v57 = (const char *)(*(__int64 (__fastcall **)(const char **, __int64))*a2)(a2, v56);
      if ( a4 != strlen(v57) )
        return 0;
      v58 = (*a3)++;
      v59 = sub_22077B0(80);
      v39 = v59;
      if ( v59 )
      {
        sub_16933F0(v59, n, v97, (_DWORD)v26, v96, v58, 0);
        v56 = v94;
      }
      while ( 1 )
      {
        v60 = *a3;
        if ( v60 >= (*((unsigned int (__fastcall **)(const char **, __int64))*a2 + 1))(a2, v56)
          || !(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3) )
        {
          break;
        }
        v56 = *a3;
        v61 = *(__int64 (__fastcall **)(const char **))*a2;
        *a3 = v56 + 1;
        v62 = v61(a2);
        v63 = *(unsigned int *)(v39 + 56);
        if ( (unsigned int)v63 >= *(_DWORD *)(v39 + 60) )
        {
          v56 = v39 + 64;
          sub_16CD150(v39 + 48, v39 + 64, 0, 8);
          v63 = *(unsigned int *)(v39 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v63) = v62;
        ++*(_DWORD *)(v39 + 56);
      }
      return v39;
    case 8:
      v64 = sub_22077B0(80);
      v39 = v64;
      if ( v64 )
      {
        sub_16933F0(v64, n, v97, (_DWORD)v26, v96, *a3, 0);
        v65 = v94;
      }
      v66 = (const char *)(*(__int64 (__fastcall **)(const char **, _QWORD, __int64))*a2)(a2, *a3, v65);
      if ( a4 != strlen(v66) )
      {
        v67 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3) + a4;
        v68 = *(unsigned int *)(v39 + 56);
        if ( (unsigned int)v68 >= *(_DWORD *)(v39 + 60) )
        {
          sub_16CD150(v39 + 48, v39 + 64, 0, 8);
          v68 = *(unsigned int *)(v39 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v68) = v67;
        ++*(_DWORD *)(v39 + 56);
      }
      v69 = *a3 + 1;
      *a3 = v69;
      while ( (*((unsigned int (__fastcall **)(const char **))*a2 + 1))(a2) > v69
           && (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3) )
      {
        v70 = *(__int64 (__fastcall **)(const char **))*a2;
        ++*a3;
        v71 = v70(a2);
        v72 = *(unsigned int *)(v39 + 56);
        if ( (unsigned int)v72 >= *(_DWORD *)(v39 + 60) )
        {
          sub_16CD150(v39 + 48, v39 + 64, 0, 8);
          v72 = *(unsigned int *)(v39 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v72) = v71;
        v69 = *a3;
        ++*(_DWORD *)(v39 + 56);
      }
      return v39;
    case 9:
      v73 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      v74 = *a3;
      v75 = (char *)(v73 + a4);
      ++*a3;
      v76 = sub_22077B0(80);
      v39 = v76;
      if ( v76 )
        sub_16933F0(v76, n, v97, (_DWORD)v26, v96, v74, 0);
      v77 = v75;
      while ( 2 )
      {
        v78 = *v77;
        if ( *v77 )
        {
          v79 = v77 + 1;
          if ( v78 != 44 )
            goto LABEL_78;
        }
        if ( v77 != v75 )
        {
          v80 = (void *)sub_2207820(v77 - v75 + 1);
          memcpy(v80, v75, v77 - v75);
          v81 = *(unsigned int *)(v39 + 56);
          *((_BYTE *)v80 + v77 - v75) = 0;
          if ( (unsigned int)v81 >= *(_DWORD *)(v39 + 60) )
          {
            sub_16CD150(v39 + 48, v39 + 64, 0, 8);
            v81 = *(unsigned int *)(v39 + 56);
          }
          *(_QWORD *)(*(_QWORD *)(v39 + 48) + 8 * v81) = v80;
          ++*(_DWORD *)(v39 + 56);
        }
        if ( v78 )
        {
          v79 = v77 + 1;
          v75 = v77 + 1;
LABEL_78:
          v77 = v79;
          continue;
        }
        break;
      }
      *(_BYTE *)(v39 + 44) |= 2u;
      return v39;
    case 0xA:
      v82 = (const char *)(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      if ( a4 != strlen(v82) )
        return 0;
      v83 = *(unsigned __int8 *)(*a1 + 37) + *a3 + 1;
      *a3 = v83;
      if ( v83 > (*((unsigned int (__fastcall **)(const char **))*a2 + 1))(a2) )
        return 0;
      v95 = *a3 - *(unsigned __int8 *)(*a1 + 37) - 1;
      v100 = (*(__int64 (__fastcall **)(const char **))*a2)(a2);
      v84 = sub_22077B0(80);
      v39 = v84;
      if ( v84 )
      {
        sub_1693430(v84, n, v97, (_DWORD)v26, v96, v95, v100, 0);
        v87 = v93;
      }
      v88 = *(unsigned __int8 *)(*a1 + 37);
      if ( v88 != 1 )
      {
        v89 = 1;
        v90 = v39 + 64;
        do
        {
          v86 = (*(__int64 (__fastcall **)(const char **, _QWORD, __int64, __int64, __int64, __int64))*a2)(
                  a2,
                  v89 + *a3 - v88,
                  v85,
                  v90,
                  v86,
                  v87);
          v92 = *(unsigned int *)(v39 + 56);
          if ( (unsigned int)v92 >= *(_DWORD *)(v39 + 60) )
          {
            v98 = v86;
            sub_16CD150(v39 + 48, v39 + 64, 0, 8);
            v92 = *(unsigned int *)(v39 + 56);
            v86 = v98;
          }
          v85 = *(_QWORD *)(v39 + 48);
          ++v89;
          *(_QWORD *)(v85 + 8 * v92) = v86;
          v91 = *a1;
          ++*(_DWORD *)(v39 + 56);
          v88 = *(unsigned __int8 *)(v91 + 37);
        }
        while ( v89 != v88 );
      }
      return v39;
    case 0xB:
      v32 = (const char *)(*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
      if ( a4 == strlen(v32) )
      {
LABEL_50:
        v54 = *a3 + 2;
        *a3 = v54;
        if ( v54 <= (*((unsigned int (__fastcall **)(const char **))*a2 + 1))(a2)
          && (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3 - 1) )
        {
          v55 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3 - 1);
          v39 = sub_22077B0(80);
          if ( v39 )
            sub_1693430(v39, n, v97, (_DWORD)v26, v96, *a3 - 2, v55, 0);
        }
        else
        {
          return 0;
        }
      }
      else
      {
        v33 = (*(__int64 (__fastcall **)(const char **, _QWORD))*a2)(a2, *a3);
        v34 = *a3;
        v35 = *a1;
        v36 = v33 + a4;
        v37 = a1[1];
        ++*a3;
        na = v34;
        v38 = sub_22077B0(80);
        v39 = v38;
        if ( v38 )
          sub_1693430(v38, v35, v37, (_DWORD)v26, v96, na, v36, 0);
      }
      return v39;
  }
}
