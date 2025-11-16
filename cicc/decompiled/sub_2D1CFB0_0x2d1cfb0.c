// Function: sub_2D1CFB0
// Address: 0x2d1cfb0
//
__int64 __fastcall sub_2D1CFB0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // esi
  __int64 v7; // rdx
  unsigned int v8; // edi
  int v9; // r11d
  __int64 *v10; // rcx
  unsigned int v11; // r8d
  __int64 *v12; // rax
  __int64 v13; // r9
  unsigned int v14; // ecx
  int v15; // r15d
  __int64 *v16; // r9
  unsigned int v17; // r8d
  __int64 *v18; // rax
  __int64 v19; // r11
  unsigned int v20; // eax
  int v22; // eax
  int v23; // esi
  __int64 v24; // r8
  unsigned int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  int v28; // eax
  int v29; // esi
  __int64 v30; // rdi
  __int64 v31; // rdx
  int v32; // eax
  __int64 v33; // r8
  __int64 v34; // r13
  __int64 v35; // r11
  __int64 v36; // r9
  unsigned int v37; // r8d
  __int64 *v38; // rax
  __int64 v39; // rdi
  int v40; // eax
  __int64 v41; // rbx
  unsigned int v42; // esi
  int v43; // edi
  int v44; // edi
  __int64 v45; // r9
  unsigned int v46; // esi
  int v47; // eax
  __int64 *v48; // rdx
  __int64 v49; // r8
  int v50; // eax
  int v51; // esi
  int v52; // esi
  __int64 v53; // r8
  __int64 *v54; // r9
  unsigned int v55; // r15d
  int v56; // r10d
  __int64 v57; // rdi
  int v58; // eax
  int v59; // eax
  int v60; // eax
  __int64 v61; // rdi
  __int64 *v62; // r8
  unsigned int v63; // r13d
  int v64; // r9d
  __int64 v65; // rsi
  int v66; // eax
  int v67; // eax
  int v68; // edx
  __int64 v69; // rdi
  __int64 *v70; // r8
  __int64 v71; // r13
  int v72; // r10d
  __int64 v73; // rsi
  int v74; // r15d
  __int64 *v75; // r10
  int v76; // r11d
  __int64 *v77; // r10
  int v78; // r10d
  __int64 *v79; // r9
  unsigned int v80; // [rsp-48h] [rbp-48h]
  unsigned int v81; // [rsp-48h] [rbp-48h]
  unsigned int v82; // [rsp-40h] [rbp-40h]
  __int64 v83; // [rsp-40h] [rbp-40h]
  int v84; // [rsp-40h] [rbp-40h]
  __int64 v85; // [rsp-40h] [rbp-40h]
  unsigned int v86; // [rsp-40h] [rbp-40h]

  if ( a2 != a3 )
  {
    v6 = *(_DWORD *)(a1 + 24);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = v6 - 1;
      v9 = 1;
      v10 = 0;
      v11 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == a2 )
      {
LABEL_4:
        v14 = *((_DWORD *)v12 + 2);
        goto LABEL_5;
      }
      while ( v13 != -4096 )
      {
        if ( !v10 && v13 == -8192 )
          v10 = v12;
        v11 = v8 & (v9 + v11);
        v12 = (__int64 *)(v7 + 16LL * v11);
        v13 = *v12;
        if ( *v12 == a2 )
          goto LABEL_4;
        ++v9;
      }
      if ( !v10 )
        v10 = v12;
      v58 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v26 = v58 + 1;
      if ( 4 * (v58 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 20) - v26 > v6 >> 3 )
        {
LABEL_11:
          *(_DWORD *)(a1 + 16) = v26;
          if ( *v10 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v10 = a2;
          *((_DWORD *)v10 + 2) = 0;
          v6 = *(_DWORD *)(a1 + 24);
          if ( !v6 )
          {
            ++*(_QWORD *)a1;
            v14 = 0;
            goto LABEL_15;
          }
          v7 = *(_QWORD *)(a1 + 8);
          v8 = v6 - 1;
          v14 = 0;
LABEL_5:
          v15 = 1;
          v16 = 0;
          v17 = v8 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
          v18 = (__int64 *)(v7 + 16LL * v17);
          v19 = *v18;
          if ( *v18 == a3 )
          {
LABEL_6:
            v20 = *((_DWORD *)v18 + 2);
            if ( v14 < v20 )
              return 1;
LABEL_20:
            if ( v14 > v20 )
              return 0;
            v34 = a2 + 24;
            v35 = *(_QWORD *)(a2 + 40) + 48LL;
            if ( a2 + 24 == v35 )
              return 0;
            while ( 1 )
            {
              v41 = v34 - 24;
              if ( !v34 )
                v41 = 0;
              if ( v41 == a3 )
                return 1;
              v42 = *(_DWORD *)(a1 + 24);
              if ( !v42 )
                break;
              v36 = *(_QWORD *)(a1 + 8);
              v37 = (v42 - 1) & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
              v38 = (__int64 *)(v36 + 16LL * v37);
              v39 = *v38;
              if ( v41 != *v38 )
              {
                v84 = 1;
                v48 = 0;
                while ( v39 != -4096 )
                {
                  if ( !v48 && v39 == -8192 )
                    v48 = v38;
                  v37 = (v42 - 1) & (v84 + v37);
                  v38 = (__int64 *)(v36 + 16LL * v37);
                  v39 = *v38;
                  if ( v41 == *v38 )
                    goto LABEL_24;
                  ++v84;
                }
                if ( !v48 )
                  v48 = v38;
                v50 = *(_DWORD *)(a1 + 16);
                ++*(_QWORD *)a1;
                v47 = v50 + 1;
                if ( 4 * v47 < 3 * v42 )
                {
                  if ( v42 - *(_DWORD *)(a1 + 20) - v47 <= v42 >> 3 )
                  {
                    v81 = v14;
                    v85 = v35;
                    sub_9BAAD0(a1, v42);
                    v51 = *(_DWORD *)(a1 + 24);
                    if ( !v51 )
                    {
LABEL_136:
                      ++*(_DWORD *)(a1 + 16);
                      BUG();
                    }
                    v52 = v51 - 1;
                    v53 = *(_QWORD *)(a1 + 8);
                    v54 = 0;
                    v55 = v52 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                    v35 = v85;
                    v14 = v81;
                    v56 = 1;
                    v47 = *(_DWORD *)(a1 + 16) + 1;
                    v48 = (__int64 *)(v53 + 16LL * v55);
                    v57 = *v48;
                    if ( v41 != *v48 )
                    {
                      while ( v57 != -4096 )
                      {
                        if ( v57 == -8192 && !v54 )
                          v54 = v48;
                        v55 = v52 & (v56 + v55);
                        v48 = (__int64 *)(v53 + 16LL * v55);
                        v57 = *v48;
                        if ( v41 == *v48 )
                          goto LABEL_34;
                        ++v56;
                      }
                      if ( v54 )
                        v48 = v54;
                    }
                  }
                  goto LABEL_34;
                }
LABEL_32:
                v80 = v14;
                v83 = v35;
                sub_9BAAD0(a1, 2 * v42);
                v43 = *(_DWORD *)(a1 + 24);
                if ( !v43 )
                  goto LABEL_136;
                v44 = v43 - 1;
                v45 = *(_QWORD *)(a1 + 8);
                v35 = v83;
                v14 = v80;
                v46 = v44 & (((unsigned int)v41 >> 9) ^ ((unsigned int)v41 >> 4));
                v47 = *(_DWORD *)(a1 + 16) + 1;
                v48 = (__int64 *)(v45 + 16LL * v46);
                v49 = *v48;
                if ( v41 != *v48 )
                {
                  v74 = 1;
                  v75 = 0;
                  while ( v49 != -4096 )
                  {
                    if ( !v75 && v49 == -8192 )
                      v75 = v48;
                    v46 = v44 & (v74 + v46);
                    v48 = (__int64 *)(v45 + 16LL * v46);
                    v49 = *v48;
                    if ( v41 == *v48 )
                      goto LABEL_34;
                    ++v74;
                  }
                  if ( v75 )
                    v48 = v75;
                }
LABEL_34:
                *(_DWORD *)(a1 + 16) = v47;
                if ( *v48 != -4096 )
                  --*(_DWORD *)(a1 + 20);
                *v48 = v41;
                v40 = 0;
                *((_DWORD *)v48 + 2) = 0;
                goto LABEL_25;
              }
LABEL_24:
              v40 = *((_DWORD *)v38 + 2);
LABEL_25:
              if ( v40 == v14 )
              {
                v34 = *(_QWORD *)(v34 + 8);
                if ( v35 != v34 )
                  continue;
              }
              return 0;
            }
            ++*(_QWORD *)a1;
            goto LABEL_32;
          }
          while ( v19 != -4096 )
          {
            if ( !v16 && v19 == -8192 )
              v16 = v18;
            v17 = v8 & (v15 + v17);
            v18 = (__int64 *)(v7 + 16LL * v17);
            v19 = *v18;
            if ( *v18 == a3 )
              goto LABEL_6;
            ++v15;
          }
          if ( !v16 )
            v16 = v18;
          v66 = *(_DWORD *)(a1 + 16);
          ++*(_QWORD *)a1;
          v32 = v66 + 1;
          if ( 4 * v32 < 3 * v6 )
          {
            if ( v6 - (v32 + *(_DWORD *)(a1 + 20)) > v6 >> 3 )
            {
LABEL_17:
              *(_DWORD *)(a1 + 16) = v32;
              if ( *v16 != -4096 )
                --*(_DWORD *)(a1 + 20);
              *v16 = a3;
              v20 = 0;
              *((_DWORD *)v16 + 2) = 0;
              goto LABEL_20;
            }
            v86 = v14;
            sub_9BAAD0(a1, v6);
            v67 = *(_DWORD *)(a1 + 24);
            if ( v67 )
            {
              v68 = v67 - 1;
              v69 = *(_QWORD *)(a1 + 8);
              v70 = 0;
              LODWORD(v71) = (v67 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
              v14 = v86;
              v72 = 1;
              v32 = *(_DWORD *)(a1 + 16) + 1;
              v16 = (__int64 *)(v69 + 16LL * (unsigned int)v71);
              v73 = *v16;
              if ( *v16 != a3 )
              {
                while ( v73 != -4096 )
                {
                  if ( v73 == -8192 && !v70 )
                    v70 = v16;
                  v71 = v68 & (unsigned int)(v71 + v72);
                  v16 = (__int64 *)(v69 + 16 * v71);
                  v73 = *v16;
                  if ( *v16 == a3 )
                    goto LABEL_17;
                  ++v72;
                }
                if ( v70 )
                  v16 = v70;
              }
              goto LABEL_17;
            }
LABEL_135:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
LABEL_15:
          v82 = v14;
          sub_9BAAD0(a1, 2 * v6);
          v28 = *(_DWORD *)(a1 + 24);
          if ( v28 )
          {
            v29 = v28 - 1;
            v30 = *(_QWORD *)(a1 + 8);
            v14 = v82;
            LODWORD(v31) = (v28 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
            v32 = *(_DWORD *)(a1 + 16) + 1;
            v16 = (__int64 *)(v30 + 16LL * (unsigned int)v31);
            v33 = *v16;
            if ( *v16 != a3 )
            {
              v76 = 1;
              v77 = 0;
              while ( v33 != -4096 )
              {
                if ( !v77 && v33 == -8192 )
                  v77 = v16;
                v31 = v29 & (unsigned int)(v31 + v76);
                v16 = (__int64 *)(v30 + 16 * v31);
                v33 = *v16;
                if ( *v16 == a3 )
                  goto LABEL_17;
                ++v76;
              }
              if ( v77 )
                v16 = v77;
            }
            goto LABEL_17;
          }
          goto LABEL_135;
        }
        sub_9BAAD0(a1, v6);
        v59 = *(_DWORD *)(a1 + 24);
        if ( v59 )
        {
          v60 = v59 - 1;
          v61 = *(_QWORD *)(a1 + 8);
          v62 = 0;
          v63 = v60 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v64 = 1;
          v26 = *(_DWORD *)(a1 + 16) + 1;
          v10 = (__int64 *)(v61 + 16LL * v63);
          v65 = *v10;
          if ( *v10 != a2 )
          {
            while ( v65 != -4096 )
            {
              if ( v65 == -8192 && !v62 )
                v62 = v10;
              v63 = v60 & (v64 + v63);
              v10 = (__int64 *)(v61 + 16LL * v63);
              v65 = *v10;
              if ( *v10 == a2 )
                goto LABEL_11;
              ++v64;
            }
            if ( v62 )
              v10 = v62;
          }
          goto LABEL_11;
        }
LABEL_134:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_9BAAD0(a1, 2 * v6);
    v22 = *(_DWORD *)(a1 + 24);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(a1 + 8);
      v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v10;
      if ( *v10 != a2 )
      {
        v78 = 1;
        v79 = 0;
        while ( v27 != -4096 )
        {
          if ( !v79 && v27 == -8192 )
            v79 = v10;
          v25 = v23 & (v78 + v25);
          v10 = (__int64 *)(v24 + 16LL * v25);
          v27 = *v10;
          if ( *v10 == a2 )
            goto LABEL_11;
          ++v78;
        }
        if ( v79 )
          v10 = v79;
      }
      goto LABEL_11;
    }
    goto LABEL_134;
  }
  return 0;
}
