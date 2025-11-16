// Function: sub_2107D70
// Address: 0x2107d70
//
__int64 __fastcall sub_2107D70(__int64 a1, __int64 a2)
{
  __int64 v2; // r11
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  unsigned int v7; // edi
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rbx
  __int64 v14; // rdx
  unsigned int v15; // r12d
  __int64 v16; // r15
  int v17; // r11d
  __int64 v18; // rax
  unsigned int v19; // esi
  int v20; // r8d
  __int64 v21; // r13
  __int64 v22; // r9
  unsigned int v23; // edi
  _QWORD *v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rax
  __int64 v27; // r13
  int v28; // eax
  unsigned int v29; // r12d
  __int64 v31; // rax
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rdx
  _QWORD *v35; // rdx
  int v36; // eax
  int v37; // eax
  __int64 v38; // rdx
  int v39; // esi
  int v40; // esi
  __int64 v41; // r9
  unsigned int v42; // ecx
  __int64 v43; // rdi
  int v44; // r14d
  _QWORD *v45; // r10
  int v46; // ecx
  int v47; // ecx
  __int64 v48; // rdi
  _QWORD *v49; // r9
  unsigned int v50; // r14d
  int v51; // r10d
  __int64 v52; // rsi
  int v53; // r10d
  _QWORD *v54; // rdx
  int v55; // eax
  int v56; // ecx
  int v57; // eax
  int v58; // esi
  __int64 v59; // rdi
  unsigned int v60; // eax
  __int64 v61; // r8
  int v62; // r10d
  _QWORD *v63; // r9
  int v64; // eax
  int v65; // eax
  __int64 v66; // rdi
  _QWORD *v67; // r8
  __int64 v68; // r13
  int v69; // r9d
  __int64 v70; // rsi
  __int64 *v71; // r14
  int v72; // [rsp+0h] [rbp-100h]
  int v73; // [rsp+0h] [rbp-100h]
  __int64 *v74; // [rsp+0h] [rbp-100h]
  int v75; // [rsp+0h] [rbp-100h]
  int v76; // [rsp+8h] [rbp-F8h]
  int v77; // [rsp+8h] [rbp-F8h]
  int v78; // [rsp+8h] [rbp-F8h]
  int v79; // [rsp+8h] [rbp-F8h]
  __int64 v80; // [rsp+8h] [rbp-F8h]
  __int64 v81; // [rsp+10h] [rbp-F0h]
  __int64 v82; // [rsp+18h] [rbp-E8h]
  __int64 v83; // [rsp+18h] [rbp-E8h]
  _QWORD *v84; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v85; // [rsp+28h] [rbp-D8h]
  _QWORD v86[26]; // [rsp+30h] [rbp-D0h] BYREF

  v2 = a1;
  v86[0] = a2;
  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_DWORD *)(a1 + 48);
  v85 = 0x1400000001LL;
  v84 = v86;
  v81 = a1 + 24;
  if ( v5 )
  {
    v6 = *(_QWORD *)(a1 + 32);
    v7 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (_QWORD *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
      v10 = v8[1];
      goto LABEL_4;
    }
    v53 = 1;
    v54 = 0;
    while ( v9 != -8 )
    {
      if ( v54 || v9 != -16 )
        v8 = v54;
      v7 = (v5 - 1) & (v53 + v7);
      v71 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v71;
      if ( v4 == *v71 )
      {
        v10 = v71[1];
LABEL_4:
        *(_QWORD *)(v10 + 56) = a2;
        v11 = v86;
        v12 = 1;
        v13 = v2;
LABEL_5:
        while ( 2 )
        {
          v14 = v12--;
          v15 = 1;
          v16 = v11[v14 - 1];
          LODWORD(v85) = v12;
          v17 = *(_DWORD *)(v16 + 40);
          if ( v17 == 1 )
            goto LABEL_23;
LABEL_6:
          v18 = *(_QWORD *)(v16 + 32);
          v19 = *(_DWORD *)(v13 + 48);
          v20 = *(_DWORD *)(v18 + 40LL * v15 + 8);
          v21 = *(_QWORD *)(v18 + 40LL * (v15 + 1) + 24);
          if ( v19 )
          {
            v22 = *(_QWORD *)(v13 + 32);
            v23 = (v19 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
            v24 = (_QWORD *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( v21 == *v24 )
            {
              v26 = v24[1];
              goto LABEL_9;
            }
            v77 = 1;
            v35 = 0;
            while ( v25 != -8 )
            {
              if ( v35 || v25 != -16 )
                v24 = v35;
              v23 = (v19 - 1) & (v77 + v23);
              v74 = (__int64 *)(v22 + 16LL * v23);
              v25 = *v74;
              if ( v21 == *v74 )
              {
                v26 = v74[1];
                goto LABEL_9;
              }
              ++v77;
              v35 = v24;
              v24 = (_QWORD *)(v22 + 16LL * v23);
            }
            if ( !v35 )
              v35 = v24;
            v36 = *(_DWORD *)(v13 + 40);
            ++*(_QWORD *)(v13 + 24);
            v37 = v36 + 1;
            if ( 4 * v37 < 3 * v19 )
            {
              if ( v19 - *(_DWORD *)(v13 + 44) - v37 <= v19 >> 3 )
              {
                v73 = v17;
                v79 = v20;
                sub_2107BB0(v81, v19);
                v46 = *(_DWORD *)(v13 + 48);
                if ( !v46 )
                {
LABEL_108:
                  v2 = v13;
LABEL_109:
                  ++*(_DWORD *)(v2 + 40);
                  BUG();
                }
                v47 = v46 - 1;
                v48 = *(_QWORD *)(v13 + 32);
                v49 = 0;
                v50 = v47 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                v20 = v79;
                v17 = v73;
                v51 = 1;
                v37 = *(_DWORD *)(v13 + 40) + 1;
                v35 = (_QWORD *)(v48 + 16LL * v50);
                v52 = *v35;
                if ( v21 != *v35 )
                {
                  while ( v52 != -8 )
                  {
                    if ( !v49 && v52 == -16 )
                      v49 = v35;
                    v50 = v47 & (v51 + v50);
                    v35 = (_QWORD *)(v48 + 16LL * v50);
                    v52 = *v35;
                    if ( v21 == *v35 )
                      goto LABEL_31;
                    ++v51;
                  }
                  if ( v49 )
                    v35 = v49;
                }
              }
LABEL_31:
              *(_DWORD *)(v13 + 40) = v37;
              if ( *v35 != -8 )
                --*(_DWORD *)(v13 + 44);
              *v35 = v21;
              v26 = 0;
              v35[1] = 0;
LABEL_9:
              v27 = *(_QWORD *)(v26 + 16);
              v28 = *(_DWORD *)(v27 + 8);
              if ( v28 )
              {
                if ( v28 != v20 )
                  goto LABEL_11;
              }
              else
              {
                v76 = v17;
                v31 = sub_1E69D00(*(_QWORD *)(*(_QWORD *)v13 + 40LL), v20);
                v17 = v76;
                if ( !v31
                  || **(_WORD **)(v31 + 16) && **(_WORD **)(v31 + 16) != 45
                  || *(_QWORD *)v27 != *(_QWORD *)(v31 + 24) )
                {
LABEL_11:
                  v11 = v84;
                  v29 = 0;
                  goto LABEL_12;
                }
                v34 = *(_QWORD *)(v27 + 56);
                if ( v34 )
                {
                  if ( v34 != v31 )
                    goto LABEL_11;
                }
                else
                {
                  *(_QWORD *)(v27 + 56) = v31;
                  v38 = (unsigned int)v85;
                  if ( (unsigned int)v85 >= HIDWORD(v85) )
                  {
                    v75 = v76;
                    v80 = v31;
                    sub_16CD150((__int64)&v84, v86, 0, 8, v32, v33);
                    v38 = (unsigned int)v85;
                    v17 = v75;
                    v31 = v80;
                  }
                  v84[v38] = v31;
                  LODWORD(v85) = v85 + 1;
                }
              }
              v15 += 2;
              if ( v17 == v15 )
              {
                v12 = v85;
                v11 = v84;
LABEL_23:
                if ( !v12 )
                {
                  v29 = 1;
                  goto LABEL_12;
                }
                continue;
              }
              goto LABEL_6;
            }
          }
          else
          {
            ++*(_QWORD *)(v13 + 24);
          }
          break;
        }
        v72 = v17;
        v78 = v20;
        sub_2107BB0(v81, 2 * v19);
        v39 = *(_DWORD *)(v13 + 48);
        if ( !v39 )
          goto LABEL_108;
        v40 = v39 - 1;
        v41 = *(_QWORD *)(v13 + 32);
        v20 = v78;
        v17 = v72;
        v42 = v40 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v37 = *(_DWORD *)(v13 + 40) + 1;
        v35 = (_QWORD *)(v41 + 16LL * v42);
        v43 = *v35;
        if ( v21 != *v35 )
        {
          v44 = 1;
          v45 = 0;
          while ( v43 != -8 )
          {
            if ( !v45 && v43 == -16 )
              v45 = v35;
            v42 = v40 & (v44 + v42);
            v35 = (_QWORD *)(v41 + 16LL * v42);
            v43 = *v35;
            if ( v21 == *v35 )
              goto LABEL_31;
            ++v44;
          }
          if ( v45 )
            v35 = v45;
        }
        goto LABEL_31;
      }
      ++v53;
      v54 = v8;
      v8 = (_QWORD *)(v6 + 16LL * v7);
    }
    if ( !v54 )
      v54 = v8;
    v55 = *(_DWORD *)(v2 + 40);
    ++*(_QWORD *)(v2 + 24);
    v56 = v55 + 1;
    if ( 4 * (v55 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(v2 + 44) - v56 <= v5 >> 3 )
      {
        v83 = v2;
        sub_2107BB0(v81, v5);
        v2 = v83;
        v64 = *(_DWORD *)(v83 + 48);
        if ( !v64 )
          goto LABEL_109;
        v65 = v64 - 1;
        v66 = *(_QWORD *)(v83 + 32);
        v67 = 0;
        LODWORD(v68) = v65 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v69 = 1;
        v56 = *(_DWORD *)(v83 + 40) + 1;
        v54 = (_QWORD *)(v66 + 16LL * (unsigned int)v68);
        v70 = *v54;
        if ( v4 != *v54 )
        {
          while ( v70 != -8 )
          {
            if ( v70 == -16 && !v67 )
              v67 = v54;
            v68 = v65 & (unsigned int)(v68 + v69);
            v54 = (_QWORD *)(v66 + 16 * v68);
            v70 = *v54;
            if ( v4 == *v54 )
              goto LABEL_57;
            ++v69;
          }
          if ( v67 )
            v54 = v67;
        }
      }
      goto LABEL_57;
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  v82 = v2;
  sub_2107BB0(v81, 2 * v5);
  v2 = v82;
  v57 = *(_DWORD *)(v82 + 48);
  if ( !v57 )
    goto LABEL_109;
  v58 = v57 - 1;
  v59 = *(_QWORD *)(v82 + 32);
  v60 = (v57 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v56 = *(_DWORD *)(v82 + 40) + 1;
  v54 = (_QWORD *)(v59 + 16LL * v60);
  v61 = *v54;
  if ( v4 != *v54 )
  {
    v62 = 1;
    v63 = 0;
    while ( v61 != -8 )
    {
      if ( !v63 && v61 == -16 )
        v63 = v54;
      v60 = v58 & (v62 + v60);
      v54 = (_QWORD *)(v59 + 16LL * v60);
      v61 = *v54;
      if ( v4 == *v54 )
        goto LABEL_57;
      ++v62;
    }
    if ( v63 )
      v54 = v63;
  }
LABEL_57:
  *(_DWORD *)(v2 + 40) = v56;
  if ( *v54 != -8 )
    --*(_DWORD *)(v2 + 44);
  *v54 = v4;
  v11 = v84;
  v54[1] = 0;
  v12 = v85;
  MEMORY[0x38] = a2;
  if ( (_DWORD)v85 )
  {
    v13 = v2;
    goto LABEL_5;
  }
  v29 = 1;
LABEL_12:
  if ( v11 != v86 )
    _libc_free((unsigned __int64)v11);
  return v29;
}
