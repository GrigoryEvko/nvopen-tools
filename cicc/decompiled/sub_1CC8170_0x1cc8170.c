// Function: sub_1CC8170
// Address: 0x1cc8170
//
__int64 __fastcall sub_1CC8170(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  unsigned int v7; // ecx
  __int64 v8; // rdx
  unsigned int v9; // r8d
  __int64 *v10; // rax
  __int64 v11; // r9
  unsigned int v12; // r11d
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // r10
  unsigned int v16; // eax
  int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  unsigned int v21; // eax
  int v22; // edx
  __int64 *v23; // rdi
  __int64 v24; // rsi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // rsi
  __int64 v28; // rdx
  int v29; // eax
  __int64 *v30; // r8
  __int64 v31; // rdi
  __int64 v32; // r13
  __int64 v33; // r10
  __int64 v34; // r8
  unsigned int v35; // edi
  __int64 *v36; // rax
  __int64 v37; // rcx
  int v38; // eax
  __int64 v39; // rbx
  unsigned int v40; // esi
  int v41; // esi
  int v42; // esi
  __int64 v43; // r8
  unsigned int v44; // ecx
  int v45; // eax
  __int64 *v46; // rdx
  __int64 v47; // rdi
  int v48; // eax
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // rdi
  __int64 *v52; // r8
  unsigned int v53; // r15d
  int v54; // r9d
  __int64 v55; // rsi
  int v56; // r11d
  int v57; // eax
  int v58; // eax
  int v59; // eax
  __int64 v60; // rsi
  __int64 *v61; // r8
  unsigned int v62; // r13d
  int v63; // r9d
  __int64 v64; // rcx
  int v65; // r15d
  int v66; // eax
  int v67; // eax
  int v68; // edx
  __int64 v69; // rsi
  __int64 *v70; // rdi
  __int64 v71; // r13
  int v72; // r9d
  __int64 v73; // rcx
  int v74; // r15d
  __int64 *v75; // r9
  int v76; // r10d
  __int64 *v77; // r9
  int v78; // r10d
  __int64 *v79; // r9
  unsigned int v80; // [rsp-48h] [rbp-48h]
  unsigned int v81; // [rsp-48h] [rbp-48h]
  __int64 *v82; // [rsp-48h] [rbp-48h]
  unsigned int v83; // [rsp-40h] [rbp-40h]
  __int64 v84; // [rsp-40h] [rbp-40h]
  int v85; // [rsp-40h] [rbp-40h]
  __int64 v86; // [rsp-40h] [rbp-40h]
  unsigned int v87; // [rsp-40h] [rbp-40h]

  if ( a2 != a3 )
  {
    v6 = *(_DWORD *)(a1 + 24);
    if ( v6 )
    {
      v7 = v6 - 1;
      v8 = *(_QWORD *)(a1 + 8);
      v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v10 == a2 )
      {
LABEL_4:
        v12 = *((_DWORD *)v10 + 2);
        goto LABEL_5;
      }
      v56 = 1;
      v23 = 0;
      while ( v11 != -8 )
      {
        if ( v11 == -16 && !v23 )
          v23 = v10;
        v9 = v7 & (v56 + v9);
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( *v10 == a2 )
          goto LABEL_4;
        ++v56;
      }
      if ( !v23 )
        v23 = v10;
      v57 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v22 = v57 + 1;
      if ( 4 * (v57 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 20) - v22 > v6 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(a1 + 16) = v22;
          if ( *v23 != -8 )
            --*(_DWORD *)(a1 + 20);
          *v23 = a2;
          *((_DWORD *)v23 + 2) = 0;
          v6 = *(_DWORD *)(a1 + 24);
          if ( !v6 )
          {
            ++*(_QWORD *)a1;
            v12 = 0;
            goto LABEL_15;
          }
          v8 = *(_QWORD *)(a1 + 8);
          v7 = v6 - 1;
          v12 = 0;
LABEL_5:
          v13 = v7 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v14 = (__int64 *)(v8 + 16LL * v13);
          v15 = *v14;
          if ( *v14 == a3 )
          {
LABEL_6:
            v16 = *((_DWORD *)v14 + 2);
            if ( v16 > v12 )
              return 1;
LABEL_20:
            if ( v12 > v16 )
              return 0;
            v32 = a2 + 24;
            v33 = *(_QWORD *)(a2 + 40) + 40LL;
            if ( v33 == a2 + 24 )
              return 0;
            while ( 1 )
            {
              v39 = v32 - 24;
              if ( !v32 )
                v39 = 0;
              if ( v39 == a3 )
                return 1;
              v40 = *(_DWORD *)(a1 + 24);
              if ( !v40 )
                break;
              v34 = *(_QWORD *)(a1 + 8);
              v35 = (v40 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
              v36 = (__int64 *)(v34 + 16LL * v35);
              v37 = *v36;
              if ( v39 != *v36 )
              {
                v85 = 1;
                v46 = 0;
                while ( v37 != -8 )
                {
                  if ( v46 || v37 != -16 )
                    v36 = v46;
                  v35 = (v40 - 1) & (v85 + v35);
                  v82 = (__int64 *)(v34 + 16LL * v35);
                  v37 = *v82;
                  if ( v39 == *v82 )
                  {
                    v38 = *((_DWORD *)v82 + 2);
                    goto LABEL_25;
                  }
                  ++v85;
                  v46 = v36;
                  v36 = (__int64 *)(v34 + 16LL * v35);
                }
                if ( !v46 )
                  v46 = v36;
                v48 = *(_DWORD *)(a1 + 16);
                ++*(_QWORD *)a1;
                v45 = v48 + 1;
                if ( 4 * v45 < 3 * v40 )
                {
                  if ( v40 - *(_DWORD *)(a1 + 20) - v45 <= v40 >> 3 )
                  {
                    v81 = v12;
                    v86 = v33;
                    sub_14672C0(a1, v40);
                    v49 = *(_DWORD *)(a1 + 24);
                    if ( !v49 )
                    {
LABEL_137:
                      ++*(_DWORD *)(a1 + 16);
                      BUG();
                    }
                    v50 = v49 - 1;
                    v51 = *(_QWORD *)(a1 + 8);
                    v52 = 0;
                    v53 = v50 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                    v33 = v86;
                    v12 = v81;
                    v54 = 1;
                    v45 = *(_DWORD *)(a1 + 16) + 1;
                    v46 = (__int64 *)(v51 + 16LL * v53);
                    v55 = *v46;
                    if ( v39 != *v46 )
                    {
                      while ( v55 != -8 )
                      {
                        if ( !v52 && v55 == -16 )
                          v52 = v46;
                        v53 = v50 & (v54 + v53);
                        v46 = (__int64 *)(v51 + 16LL * v53);
                        v55 = *v46;
                        if ( v39 == *v46 )
                          goto LABEL_34;
                        ++v54;
                      }
                      if ( v52 )
                        v46 = v52;
                    }
                  }
                  goto LABEL_34;
                }
LABEL_32:
                v80 = v12;
                v84 = v33;
                sub_14672C0(a1, 2 * v40);
                v41 = *(_DWORD *)(a1 + 24);
                if ( !v41 )
                  goto LABEL_137;
                v42 = v41 - 1;
                v43 = *(_QWORD *)(a1 + 8);
                v33 = v84;
                v12 = v80;
                v44 = v42 & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
                v45 = *(_DWORD *)(a1 + 16) + 1;
                v46 = (__int64 *)(v43 + 16LL * v44);
                v47 = *v46;
                if ( v39 != *v46 )
                {
                  v74 = 1;
                  v75 = 0;
                  while ( v47 != -8 )
                  {
                    if ( !v75 && v47 == -16 )
                      v75 = v46;
                    v44 = v42 & (v74 + v44);
                    v46 = (__int64 *)(v43 + 16LL * v44);
                    v47 = *v46;
                    if ( v39 == *v46 )
                      goto LABEL_34;
                    ++v74;
                  }
                  if ( v75 )
                    v46 = v75;
                }
LABEL_34:
                *(_DWORD *)(a1 + 16) = v45;
                if ( *v46 != -8 )
                  --*(_DWORD *)(a1 + 20);
                *v46 = v39;
                v38 = 0;
                *((_DWORD *)v46 + 2) = 0;
                goto LABEL_25;
              }
              v38 = *((_DWORD *)v36 + 2);
LABEL_25:
              if ( v38 == v12 )
              {
                v32 = *(_QWORD *)(v32 + 8);
                if ( v33 != v32 )
                  continue;
              }
              return 0;
            }
            ++*(_QWORD *)a1;
            goto LABEL_32;
          }
          v65 = 1;
          v30 = 0;
          while ( v15 != -8 )
          {
            if ( v15 == -16 && !v30 )
              v30 = v14;
            v13 = v7 & (v65 + v13);
            v14 = (__int64 *)(v8 + 16LL * v13);
            v15 = *v14;
            if ( *v14 == a3 )
              goto LABEL_6;
            ++v65;
          }
          if ( !v30 )
            v30 = v14;
          v66 = *(_DWORD *)(a1 + 16);
          ++*(_QWORD *)a1;
          v29 = v66 + 1;
          if ( 4 * v29 < 3 * v6 )
          {
            if ( v6 - (v29 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
            {
LABEL_17:
              *(_DWORD *)(a1 + 16) = v29;
              if ( *v30 != -8 )
                --*(_DWORD *)(a1 + 20);
              *v30 = a3;
              v16 = 0;
              *((_DWORD *)v30 + 2) = 0;
              goto LABEL_20;
            }
            v87 = v12;
            sub_14672C0(a1, v6);
            v67 = *(_DWORD *)(a1 + 24);
            if ( v67 )
            {
              v68 = v67 - 1;
              v69 = *(_QWORD *)(a1 + 8);
              v70 = 0;
              LODWORD(v71) = (v67 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
              v12 = v87;
              v72 = 1;
              v29 = *(_DWORD *)(a1 + 16) + 1;
              v30 = (__int64 *)(v69 + 16LL * (unsigned int)v71);
              v73 = *v30;
              if ( *v30 != a3 )
              {
                while ( v73 != -8 )
                {
                  if ( v73 == -16 && !v70 )
                    v70 = v30;
                  v71 = v68 & (unsigned int)(v71 + v72);
                  v30 = (__int64 *)(v69 + 16 * v71);
                  v73 = *v30;
                  if ( *v30 == a3 )
                    goto LABEL_17;
                  ++v72;
                }
                if ( v70 )
                  v30 = v70;
              }
              goto LABEL_17;
            }
LABEL_138:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
LABEL_15:
          v83 = v12;
          sub_14672C0(a1, 2 * v6);
          v25 = *(_DWORD *)(a1 + 24);
          if ( v25 )
          {
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 8);
            v12 = v83;
            LODWORD(v28) = (v25 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
            v29 = *(_DWORD *)(a1 + 16) + 1;
            v30 = (__int64 *)(v27 + 16LL * (unsigned int)v28);
            v31 = *v30;
            if ( *v30 != a3 )
            {
              v78 = 1;
              v79 = 0;
              while ( v31 != -8 )
              {
                if ( !v79 && v31 == -16 )
                  v79 = v30;
                v28 = v26 & (unsigned int)(v28 + v78);
                v30 = (__int64 *)(v27 + 16 * v28);
                v31 = *v30;
                if ( *v30 == a3 )
                  goto LABEL_17;
                ++v78;
              }
              if ( v79 )
                v30 = v79;
            }
            goto LABEL_17;
          }
          goto LABEL_138;
        }
        sub_14672C0(a1, v6);
        v58 = *(_DWORD *)(a1 + 24);
        if ( v58 )
        {
          v59 = v58 - 1;
          v60 = *(_QWORD *)(a1 + 8);
          v61 = 0;
          v62 = v59 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v63 = 1;
          v22 = *(_DWORD *)(a1 + 16) + 1;
          v23 = (__int64 *)(v60 + 16LL * v62);
          v64 = *v23;
          if ( *v23 != a2 )
          {
            while ( v64 != -8 )
            {
              if ( v64 == -16 && !v61 )
                v61 = v23;
              v62 = v59 & (v63 + v62);
              v23 = (__int64 *)(v60 + 16LL * v62);
              v64 = *v23;
              if ( *v23 == a2 )
                goto LABEL_11;
              ++v63;
            }
            if ( v61 )
              v23 = v61;
          }
          goto LABEL_11;
        }
LABEL_139:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_14672C0(a1, 2 * v6);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(a1 + 8);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(a1 + 16) + 1;
      v23 = (__int64 *)(v20 + 16LL * v21);
      v24 = *v23;
      if ( *v23 != a2 )
      {
        v76 = 1;
        v77 = 0;
        while ( v24 != -8 )
        {
          if ( !v77 && v24 == -16 )
            v77 = v23;
          v21 = v19 & (v76 + v21);
          v23 = (__int64 *)(v20 + 16LL * v21);
          v24 = *v23;
          if ( *v23 == a2 )
            goto LABEL_11;
          ++v76;
        }
        if ( v77 )
          v23 = v77;
      }
      goto LABEL_11;
    }
    goto LABEL_139;
  }
  return 0;
}
