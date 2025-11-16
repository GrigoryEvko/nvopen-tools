// Function: sub_31B3E50
// Address: 0x31b3e50
//
char __fastcall sub_31B3E50(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r11
  __int64 v4; // r10
  __int64 *v5; // r15
  __int64 *v6; // r12
  __int64 v8; // rdi
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // esi
  __int64 v13; // r13
  int v14; // ecx
  int v15; // ecx
  __int64 v16; // rdi
  unsigned int v17; // edx
  _QWORD *v18; // r9
  __int64 v19; // rsi
  int v20; // eax
  int v21; // r14d
  _QWORD *v22; // r8
  __int64 v23; // rax
  __int64 *v24; // rdx
  __int64 *v25; // r14
  __int64 *v26; // r13
  __int64 v27; // r8
  unsigned int v28; // edi
  __int64 *v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // r12
  unsigned int v32; // esi
  int v33; // eax
  int v34; // edx
  __int64 v35; // rdi
  __int64 *v36; // r10
  __int64 v37; // rsi
  int v38; // ecx
  int v39; // r9d
  __int64 *v40; // r8
  int v41; // eax
  __int64 *v42; // rdx
  __int64 *v43; // r14
  __int64 *v44; // r13
  __int64 v45; // r8
  unsigned int v46; // edi
  __int64 *v47; // rcx
  __int64 v48; // rdx
  __int64 v49; // r12
  unsigned int v50; // esi
  int v51; // eax
  int v52; // edx
  __int64 v53; // rdi
  __int64 *v54; // r10
  __int64 v55; // rsi
  int v56; // ecx
  int v57; // r9d
  __int64 *v58; // r8
  int v59; // edx
  int v60; // edx
  __int64 v61; // rsi
  _QWORD *v62; // rdi
  unsigned int v63; // r14d
  int v64; // r8d
  __int64 v65; // rcx
  int v66; // r11d
  int v67; // edi
  int v68; // r11d
  int v69; // edi
  int v70; // edx
  int v71; // edx
  __int64 v72; // rdi
  int v73; // r9d
  __int64 v74; // rsi
  int v75; // edx
  int v76; // edx
  __int64 v77; // rdi
  int v78; // r9d
  __int64 v79; // rsi
  __int64 v81; // [rsp+8h] [rbp-48h]
  __int64 v82; // [rsp+8h] [rbp-48h]
  __int64 v83; // [rsp+10h] [rbp-40h]
  int v84; // [rsp+10h] [rbp-40h]
  __int64 v85; // [rsp+10h] [rbp-40h]
  __int64 v86; // [rsp+18h] [rbp-38h]
  unsigned int v87; // [rsp+18h] [rbp-38h]
  unsigned int v88; // [rsp+18h] [rbp-38h]

  v3 = (__int64)a2;
  v4 = a3;
  v5 = &a2[a3];
  v6 = a2;
  v86 = a1 + 56;
  if ( a2 != v5 )
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 80);
      v13 = *v6;
      if ( !v12 )
        break;
      v8 = *(_QWORD *)(a1 + 64);
      v9 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (_QWORD *)(v8 + 8LL * v9);
      v11 = *v10;
      if ( v13 != *v10 )
      {
        v84 = 1;
        v18 = 0;
        while ( v11 != -4096 )
        {
          if ( v11 != -8192 || v18 )
            v10 = v18;
          v9 = (v12 - 1) & (v84 + v9);
          v11 = *(_QWORD *)(v8 + 8LL * v9);
          if ( v13 == v11 )
            goto LABEL_4;
          ++v84;
          v18 = v10;
          v10 = (_QWORD *)(v8 + 8LL * v9);
        }
        v41 = *(_DWORD *)(a1 + 72);
        if ( !v18 )
          v18 = v10;
        ++*(_QWORD *)(a1 + 56);
        v20 = v41 + 1;
        if ( 4 * v20 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 76) - v20 <= v12 >> 3 )
          {
            v82 = v4;
            v85 = v3;
            sub_31B3C80(v86, v12);
            v59 = *(_DWORD *)(a1 + 80);
            if ( !v59 )
            {
LABEL_131:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
            v60 = v59 - 1;
            v61 = *(_QWORD *)(a1 + 64);
            v62 = 0;
            v3 = v85;
            v63 = v60 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v4 = v82;
            v64 = 1;
            v18 = (_QWORD *)(v61 + 8LL * v63);
            v65 = *v18;
            v20 = *(_DWORD *)(a1 + 72) + 1;
            if ( v13 != *v18 )
            {
              while ( v65 != -4096 )
              {
                if ( !v62 && v65 == -8192 )
                  v62 = v18;
                v63 = v60 & (v64 + v63);
                v18 = (_QWORD *)(v61 + 8LL * v63);
                v65 = *v18;
                if ( v13 == *v18 )
                  goto LABEL_37;
                ++v64;
              }
              if ( v62 )
                v18 = v62;
            }
          }
          goto LABEL_37;
        }
LABEL_7:
        v81 = v4;
        v83 = v3;
        sub_31B3C80(v86, 2 * v12);
        v14 = *(_DWORD *)(a1 + 80);
        if ( !v14 )
          goto LABEL_131;
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 64);
        v3 = v83;
        v4 = v81;
        v17 = v15 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
        v18 = (_QWORD *)(v16 + 8LL * v17);
        v19 = *v18;
        v20 = *(_DWORD *)(a1 + 72) + 1;
        if ( v13 != *v18 )
        {
          v21 = 1;
          v22 = 0;
          while ( v19 != -4096 )
          {
            if ( v19 == -8192 && !v22 )
              v22 = v18;
            v17 = v15 & (v21 + v17);
            v18 = (_QWORD *)(v16 + 8LL * v17);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_37;
            ++v21;
          }
          if ( v22 )
            v18 = v22;
        }
LABEL_37:
        *(_DWORD *)(a1 + 72) = v20;
        if ( *v18 != -4096 )
          --*(_DWORD *)(a1 + 76);
        *v18 = v13;
      }
LABEL_4:
      if ( v5 == ++v6 )
        goto LABEL_14;
    }
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_7;
  }
LABEL_14:
  LODWORD(v23) = *(_DWORD *)(*(_QWORD *)v3 + 32LL);
  if ( (_DWORD)v23 == 11 )
  {
    v23 = sub_31AFAB0(v3, v4, 1);
    v43 = v42;
    v44 = (__int64 *)v23;
    if ( (__int64 *)v23 == v42 )
      return v23;
    while ( 1 )
    {
      v49 = sub_318B650(*v44);
      LOBYTE(v23) = sub_318B630(v49);
      if ( v49 )
      {
        if ( (_BYTE)v23 )
        {
          v50 = *(_DWORD *)(a1 + 80);
          if ( !v50 )
          {
            ++*(_QWORD *)(a1 + 56);
            goto LABEL_48;
          }
          v45 = *(_QWORD *)(a1 + 64);
          LODWORD(v23) = ((unsigned int)v49 >> 4) ^ ((unsigned int)v49 >> 9);
          v46 = (v50 - 1) & v23;
          v47 = (__int64 *)(v45 + 8LL * v46);
          v48 = *v47;
          if ( v49 != *v47 )
            break;
        }
      }
LABEL_43:
      if ( v43 == ++v44 )
        return v23;
    }
    v66 = 1;
    v54 = 0;
    while ( v48 != -4096 )
    {
      if ( v48 != -8192 || v54 )
        v47 = v54;
      v46 = (v50 - 1) & (v66 + v46);
      v48 = *(_QWORD *)(v45 + 8LL * v46);
      if ( v49 == v48 )
        goto LABEL_43;
      ++v66;
      v54 = v47;
      v47 = (__int64 *)(v45 + 8LL * v46);
    }
    v67 = *(_DWORD *)(a1 + 72);
    if ( !v54 )
      v54 = v47;
    ++*(_QWORD *)(a1 + 56);
    v56 = v67 + 1;
    if ( 4 * (v67 + 1) >= 3 * v50 )
    {
LABEL_48:
      sub_31B3C80(a1 + 56, 2 * v50);
      v51 = *(_DWORD *)(a1 + 80);
      if ( !v51 )
        goto LABEL_130;
      v52 = v51 - 1;
      v53 = *(_QWORD *)(a1 + 64);
      LODWORD(v23) = (v51 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
      v54 = (__int64 *)(v53 + 8LL * (unsigned int)v23);
      v55 = *v54;
      v56 = *(_DWORD *)(a1 + 72) + 1;
      if ( v49 == *v54 )
        goto LABEL_71;
      v57 = 1;
      v58 = 0;
      while ( v55 != -4096 )
      {
        if ( v55 == -8192 && !v58 )
          v58 = v54;
        LODWORD(v23) = v52 & (v57 + v23);
        v54 = (__int64 *)(v53 + 8LL * (unsigned int)v23);
        v55 = *v54;
        if ( v49 == *v54 )
          goto LABEL_71;
        ++v57;
      }
    }
    else
    {
      if ( v50 - *(_DWORD *)(a1 + 76) - v56 > v50 >> 3 )
        goto LABEL_71;
      v87 = ((unsigned int)v49 >> 4) ^ ((unsigned int)v49 >> 9);
      sub_31B3C80(a1 + 56, v50);
      v70 = *(_DWORD *)(a1 + 80);
      if ( !v70 )
      {
LABEL_130:
        ++*(_DWORD *)(a1 + 72);
        BUG();
      }
      v71 = v70 - 1;
      v72 = *(_QWORD *)(a1 + 64);
      v73 = 1;
      v58 = 0;
      LODWORD(v23) = v71 & v87;
      v54 = (__int64 *)(v72 + 8LL * (v71 & v87));
      v74 = *v54;
      v56 = *(_DWORD *)(a1 + 72) + 1;
      if ( v49 == *v54 )
        goto LABEL_71;
      while ( v74 != -4096 )
      {
        if ( !v58 && v74 == -8192 )
          v58 = v54;
        LODWORD(v23) = v71 & (v73 + v23);
        v54 = (__int64 *)(v72 + 8LL * (unsigned int)v23);
        v74 = *v54;
        if ( v49 == *v54 )
          goto LABEL_71;
        ++v73;
      }
    }
    if ( v58 )
      v54 = v58;
LABEL_71:
    *(_DWORD *)(a1 + 72) = v56;
    if ( *v54 != -4096 )
      --*(_DWORD *)(a1 + 76);
    *v54 = v49;
    goto LABEL_43;
  }
  if ( (_DWORD)v23 == 12 )
  {
    v23 = sub_31AFAB0(v3, v4, 1);
    v25 = v24;
    v26 = (__int64 *)v23;
    if ( v24 != (__int64 *)v23 )
    {
      while ( 1 )
      {
        v31 = sub_318B6A0(*v26);
        LOBYTE(v23) = sub_318B630(v31);
        if ( v31 )
        {
          if ( (_BYTE)v23 )
          {
            v32 = *(_DWORD *)(a1 + 80);
            if ( !v32 )
            {
              ++*(_QWORD *)(a1 + 56);
              goto LABEL_24;
            }
            v27 = *(_QWORD *)(a1 + 64);
            LODWORD(v23) = ((unsigned int)v31 >> 4) ^ ((unsigned int)v31 >> 9);
            v28 = (v32 - 1) & v23;
            v29 = (__int64 *)(v27 + 8LL * v28);
            v30 = *v29;
            if ( v31 != *v29 )
              break;
          }
        }
LABEL_19:
        if ( v25 == ++v26 )
          return v23;
      }
      v68 = 1;
      v36 = 0;
      while ( v30 != -4096 )
      {
        if ( v36 || v30 != -8192 )
          v29 = v36;
        v28 = (v32 - 1) & (v68 + v28);
        v30 = *(_QWORD *)(v27 + 8LL * v28);
        if ( v31 == v30 )
          goto LABEL_19;
        ++v68;
        v36 = v29;
        v29 = (__int64 *)(v27 + 8LL * v28);
      }
      v69 = *(_DWORD *)(a1 + 72);
      if ( !v36 )
        v36 = v29;
      ++*(_QWORD *)(a1 + 56);
      v38 = v69 + 1;
      if ( 4 * (v69 + 1) >= 3 * v32 )
      {
LABEL_24:
        sub_31B3C80(a1 + 56, 2 * v32);
        v33 = *(_DWORD *)(a1 + 80);
        if ( !v33 )
          goto LABEL_132;
        v34 = v33 - 1;
        v35 = *(_QWORD *)(a1 + 64);
        LODWORD(v23) = (v33 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
        v36 = (__int64 *)(v35 + 8LL * (unsigned int)v23);
        v37 = *v36;
        v38 = *(_DWORD *)(a1 + 72) + 1;
        if ( v31 == *v36 )
          goto LABEL_80;
        v39 = 1;
        v40 = 0;
        while ( v37 != -4096 )
        {
          if ( v37 == -8192 && !v40 )
            v40 = v36;
          LODWORD(v23) = v34 & (v39 + v23);
          v36 = (__int64 *)(v35 + 8LL * (unsigned int)v23);
          v37 = *v36;
          if ( v31 == *v36 )
            goto LABEL_80;
          ++v39;
        }
      }
      else
      {
        if ( v32 - *(_DWORD *)(a1 + 76) - v38 > v32 >> 3 )
          goto LABEL_80;
        v88 = ((unsigned int)v31 >> 4) ^ ((unsigned int)v31 >> 9);
        sub_31B3C80(a1 + 56, v32);
        v75 = *(_DWORD *)(a1 + 80);
        if ( !v75 )
        {
LABEL_132:
          ++*(_DWORD *)(a1 + 72);
          BUG();
        }
        v76 = v75 - 1;
        v77 = *(_QWORD *)(a1 + 64);
        v78 = 1;
        v40 = 0;
        LODWORD(v23) = v76 & v88;
        v36 = (__int64 *)(v77 + 8LL * (v76 & v88));
        v79 = *v36;
        v38 = *(_DWORD *)(a1 + 72) + 1;
        if ( v31 == *v36 )
          goto LABEL_80;
        while ( v79 != -4096 )
        {
          if ( !v40 && v79 == -8192 )
            v40 = v36;
          LODWORD(v23) = v76 & (v78 + v23);
          v36 = (__int64 *)(v77 + 8LL * (unsigned int)v23);
          v79 = *v36;
          if ( v31 == *v36 )
            goto LABEL_80;
          ++v78;
        }
      }
      if ( v40 )
        v36 = v40;
LABEL_80:
      *(_DWORD *)(a1 + 72) = v38;
      if ( *v36 != -4096 )
        --*(_DWORD *)(a1 + 76);
      *v36 = v31;
      goto LABEL_19;
    }
  }
  return v23;
}
