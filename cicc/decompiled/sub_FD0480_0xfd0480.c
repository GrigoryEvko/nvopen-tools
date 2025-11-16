// Function: sub_FD0480
// Address: 0xfd0480
//
char __fastcall sub_FD0480(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rax
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r14d
  __int64 *v10; // r9
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r11
  unsigned int v14; // ecx
  __int64 v15; // rdx
  __int64 v16; // rax
  unsigned int v17; // r8d
  int v18; // r10d
  unsigned int v19; // r12d
  unsigned int v20; // edi
  unsigned int v21; // r14d
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r9
  unsigned int v25; // ecx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // r11d
  __int64 v29; // r14
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rcx
  _BYTE *v33; // r12
  unsigned int v34; // r8d
  unsigned int v35; // r15d
  unsigned int v36; // edi
  __int64 v37; // rax
  _BYTE *v38; // rdx
  int v39; // r11d
  __int64 v40; // r9
  unsigned int v41; // ecx
  __int64 v42; // rdx
  __int64 v43; // rax
  int v44; // eax
  int v45; // eax
  _QWORD *v46; // r9
  _QWORD *v47; // r10
  _BYTE *v48; // r9
  unsigned int v49; // r10d
  int v50; // edx
  int v51; // edx
  __int64 v52; // rsi
  __int64 v53; // r15
  _BYTE *v54; // rcx
  int v55; // r8d
  __int64 v56; // rdi
  int v57; // eax
  int v58; // edx
  int v59; // r10d
  __int64 *v60; // r9
  __int64 v61; // rdi
  int v62; // eax
  int v63; // eax
  int v64; // edx
  int v65; // edx
  __int64 v66; // rsi
  int v67; // r8d
  __int64 v68; // r15
  _BYTE *v69; // rcx
  int v70; // eax
  int v71; // ecx
  __int64 v72; // rdi
  unsigned int v73; // eax
  __int64 v74; // rsi
  int v75; // r10d
  __int64 *v76; // r8
  int v77; // eax
  int v78; // eax
  __int64 v79; // rsi
  int v80; // r8d
  unsigned int v81; // r12d
  __int64 *v82; // rdi
  __int64 v83; // rcx
  int v84; // ecx
  int v85; // ecx
  __int64 v86; // rdi
  int v87; // eax
  __int64 *v88; // rdx
  unsigned int v89; // r12d
  __int64 v90; // rsi
  int v91; // ecx
  int v92; // ecx
  __int64 v93; // rdi
  unsigned int v94; // r12d
  __int64 v95; // rsi
  int v96; // eax
  int v98; // [rsp+Ch] [rbp-44h]
  __int64 v99; // [rsp+10h] [rbp-40h]

  LOBYTE(v5) = sub_FCD520(a2);
  if ( (_BYTE)v5 )
  {
    v6 = *(_QWORD *)(a1 + 64);
    v7 = *(_DWORD *)(a1 + 80);
    if ( *(_BYTE *)a2 == 84 )
    {
      v8 = a1 + 56;
      if ( v7 )
      {
        v9 = 1;
        v10 = 0;
        v11 = (v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v12 = (__int64 *)(v6 + 16LL * v11);
        v13 = *v12;
        if ( a2 == *v12 )
        {
LABEL_5:
          v14 = *((_DWORD *)v12 + 2);
          v15 = 1LL << v14;
          v16 = 8LL * (v14 >> 6);
LABEL_6:
          v5 = (_QWORD *)(*(_QWORD *)(a3 + 24) + v16);
          *v5 |= v15;
          return (char)v5;
        }
        while ( v13 != -4096 )
        {
          if ( v13 == -8192 && !v10 )
            v10 = v12;
          v11 = (v7 - 1) & (v9 + v11);
          v12 = (__int64 *)(v6 + 16LL * v11);
          v13 = *v12;
          if ( a2 == *v12 )
            goto LABEL_5;
          ++v9;
        }
        if ( !v10 )
          v10 = v12;
        v57 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v58 = v57 + 1;
        if ( 4 * (v57 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a1 + 76) - v58 > v7 >> 3 )
          {
LABEL_68:
            *(_DWORD *)(a1 + 72) = v58;
            if ( *v10 != -4096 )
              --*(_DWORD *)(a1 + 76);
            *v10 = a2;
            v15 = 1;
            v16 = 0;
            *((_DWORD *)v10 + 2) = 0;
            goto LABEL_6;
          }
          sub_CE2410(v8, v7);
          v77 = *(_DWORD *)(a1 + 80);
          if ( v77 )
          {
            v78 = v77 - 1;
            v79 = *(_QWORD *)(a1 + 64);
            v80 = 1;
            v81 = v78 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v58 = *(_DWORD *)(a1 + 72) + 1;
            v82 = 0;
            v10 = (__int64 *)(v79 + 16LL * v81);
            v83 = *v10;
            if ( a2 != *v10 )
            {
              while ( v83 != -4096 )
              {
                if ( v83 == -8192 && !v82 )
                  v82 = v10;
                v81 = v78 & (v80 + v81);
                v10 = (__int64 *)(v79 + 16LL * v81);
                v83 = *v10;
                if ( a2 == *v10 )
                  goto LABEL_68;
                ++v80;
              }
              if ( v82 )
                v10 = v82;
            }
            goto LABEL_68;
          }
LABEL_146:
          ++*(_DWORD *)(a1 + 72);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 56);
      }
      sub_CE2410(v8, 2 * v7);
      v70 = *(_DWORD *)(a1 + 80);
      if ( v70 )
      {
        v71 = v70 - 1;
        v72 = *(_QWORD *)(a1 + 64);
        v73 = (v70 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v58 = *(_DWORD *)(a1 + 72) + 1;
        v10 = (__int64 *)(v72 + 16LL * v73);
        v74 = *v10;
        if ( a2 != *v10 )
        {
          v75 = 1;
          v76 = 0;
          while ( v74 != -4096 )
          {
            if ( !v76 && v74 == -8192 )
              v76 = v10;
            v73 = v71 & (v75 + v73);
            v10 = (__int64 *)(v72 + 16LL * v73);
            v74 = *v10;
            if ( a2 == *v10 )
              goto LABEL_68;
            ++v75;
          }
          if ( v76 )
            v10 = v76;
        }
        goto LABEL_68;
      }
      goto LABEL_146;
    }
    if ( !v7 )
      goto LABEL_12;
    v17 = v7 - 1;
    v18 = 1;
    v19 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v20 = (v7 - 1) & v19;
    v21 = v20;
    v22 = (__int64 *)(v6 + 16LL * v20);
    v23 = *v22;
    v24 = *v22;
    if ( a2 == *v22 )
    {
LABEL_10:
      v25 = *((_DWORD *)v22 + 2);
      v26 = ~(1LL << v25);
      v27 = 8LL * (v25 >> 6);
LABEL_11:
      *(_QWORD *)(*(_QWORD *)(a3 + 24) + v27) &= v26;
LABEL_12:
      v28 = *(_DWORD *)(a2 + 4);
      LODWORD(v5) = v28 & 0x7FFFFFF;
      if ( (v28 & 0x7FFFFFF) == 0 )
        return (char)v5;
      v29 = 0;
      v99 = a1 + 56;
      while ( 1 )
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v30 = *(_QWORD *)(a2 - 8);
        else
          v30 = a2 - 32LL * (unsigned int)v5;
        v31 = *(_DWORD *)(a1 + 80);
        v32 = *(_QWORD *)(a1 + 64);
        v33 = *(_BYTE **)(v30 + 32 * v29);
        if ( !v31 )
          goto LABEL_22;
        v34 = v31 - 1;
        v35 = ((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4);
        v36 = (v31 - 1) & v35;
        v37 = v32 + 16LL * v36;
        v38 = *(_BYTE **)v37;
        if ( v33 != *(_BYTE **)v37 )
        {
          v98 = 1;
          v48 = *(_BYTE **)v37;
          v49 = (v31 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
          do
          {
            if ( v48 == (_BYTE *)-4096LL )
              goto LABEL_22;
            v49 = v34 & (v98 + v49);
            ++v98;
            v48 = *(_BYTE **)(v32 + 16LL * v49);
          }
          while ( v33 != v48 );
          v37 = v32 + 16LL * (v34 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4)));
        }
        if ( *v33 == 22 )
        {
          if ( *(_BYTE *)(a1 + 204) )
          {
            v46 = *(_QWORD **)(a1 + 184);
            v47 = &v46[*(unsigned int *)(a1 + 196)];
            if ( v46 != v47 )
            {
              while ( v33 != (_BYTE *)*v46 )
              {
                if ( v47 == ++v46 )
                  goto LABEL_18;
              }
              goto LABEL_22;
            }
          }
          else
          {
            if ( sub_C8CA60(a1 + 176, (__int64)v33) )
              goto LABEL_21;
            v31 = *(_DWORD *)(a1 + 80);
            if ( !v31 )
            {
              ++*(_QWORD *)(a1 + 56);
              goto LABEL_51;
            }
            v34 = v31 - 1;
            v32 = *(_QWORD *)(a1 + 64);
            v36 = (v31 - 1) & v35;
            v37 = v32 + 16LL * v36;
            v38 = *(_BYTE **)v37;
          }
        }
LABEL_18:
        v39 = 1;
        v40 = 0;
        if ( v33 != v38 )
          break;
LABEL_19:
        v41 = *(_DWORD *)(v37 + 8);
        v42 = 1LL << v41;
        v43 = 8LL * (v41 >> 6);
LABEL_20:
        *(_QWORD *)(*(_QWORD *)(a3 + 24) + v43) |= v42;
LABEL_21:
        v28 = *(_DWORD *)(a2 + 4);
LABEL_22:
        ++v29;
        LODWORD(v5) = v28 & 0x7FFFFFF;
        if ( (v28 & 0x7FFFFFFu) <= (unsigned int)v29 )
          return (char)v5;
      }
      while ( v38 != (_BYTE *)-4096LL )
      {
        if ( !v40 && v38 == (_BYTE *)-8192LL )
          v40 = v37;
        v36 = v34 & (v39 + v36);
        v37 = v32 + 16LL * v36;
        v38 = *(_BYTE **)v37;
        if ( v33 == *(_BYTE **)v37 )
          goto LABEL_19;
        ++v39;
      }
      if ( !v40 )
        v40 = v37;
      v44 = *(_DWORD *)(a1 + 72);
      ++*(_QWORD *)(a1 + 56);
      v45 = v44 + 1;
      if ( 4 * v45 >= 3 * v31 )
      {
LABEL_51:
        sub_CE2410(v99, 2 * v31);
        v50 = *(_DWORD *)(a1 + 80);
        if ( !v50 )
          goto LABEL_148;
        v51 = v50 - 1;
        v52 = *(_QWORD *)(a1 + 64);
        LODWORD(v53) = v51 & v35;
        v45 = *(_DWORD *)(a1 + 72) + 1;
        v40 = v52 + 16LL * (unsigned int)v53;
        v54 = *(_BYTE **)v40;
        if ( v33 == *(_BYTE **)v40 )
          goto LABEL_35;
        v55 = 1;
        v56 = 0;
        while ( v54 != (_BYTE *)-4096LL )
        {
          if ( !v56 && v54 == (_BYTE *)-8192LL )
            v56 = v40;
          v53 = v51 & (unsigned int)(v53 + v55);
          v40 = v52 + 16 * v53;
          v54 = *(_BYTE **)v40;
          if ( v33 == *(_BYTE **)v40 )
            goto LABEL_35;
          ++v55;
        }
      }
      else
      {
        if ( v31 - (v45 + *(_DWORD *)(a1 + 76)) > v31 >> 3 )
          goto LABEL_35;
        sub_CE2410(v99, v31);
        v64 = *(_DWORD *)(a1 + 80);
        if ( !v64 )
        {
LABEL_148:
          ++*(_DWORD *)(a1 + 72);
          BUG();
        }
        v65 = v64 - 1;
        v66 = *(_QWORD *)(a1 + 64);
        v67 = 1;
        LODWORD(v68) = v65 & v35;
        v56 = 0;
        v45 = *(_DWORD *)(a1 + 72) + 1;
        v40 = v66 + 16LL * (unsigned int)v68;
        v69 = *(_BYTE **)v40;
        if ( v33 == *(_BYTE **)v40 )
          goto LABEL_35;
        while ( v69 != (_BYTE *)-4096LL )
        {
          if ( v69 == (_BYTE *)-8192LL && !v56 )
            v56 = v40;
          v68 = v65 & (unsigned int)(v68 + v67);
          v40 = v66 + 16 * v68;
          v69 = *(_BYTE **)v40;
          if ( v33 == *(_BYTE **)v40 )
            goto LABEL_35;
          ++v67;
        }
      }
      if ( v56 )
        v40 = v56;
LABEL_35:
      *(_DWORD *)(a1 + 72) = v45;
      if ( *(_QWORD *)v40 != -4096 )
        --*(_DWORD *)(a1 + 76);
      *(_QWORD *)v40 = v33;
      v42 = 1;
      v43 = 0;
      *(_DWORD *)(v40 + 8) = 0;
      goto LABEL_20;
    }
    while ( 1 )
    {
      if ( v24 == -4096 )
        goto LABEL_12;
      v21 = v17 & (v18 + v21);
      v24 = *(_QWORD *)(v6 + 16LL * v21);
      if ( a2 == v24 )
        break;
      ++v18;
    }
    v59 = 1;
    v60 = 0;
    while ( v23 != -4096 )
    {
      if ( v23 == -8192 && !v60 )
        v60 = v22;
      v20 = v17 & (v59 + v20);
      v22 = (__int64 *)(v6 + 16LL * v20);
      v23 = *v22;
      if ( a2 == *v22 )
        goto LABEL_10;
      ++v59;
    }
    v61 = a1 + 56;
    if ( !v60 )
      v60 = v22;
    v62 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v63 = v62 + 1;
    if ( 4 * v63 >= 3 * v7 )
    {
      sub_CE2410(v61, 2 * v7);
      v84 = *(_DWORD *)(a1 + 80);
      if ( v84 )
      {
        v85 = v84 - 1;
        v86 = *(_QWORD *)(a1 + 64);
        v87 = 1;
        v88 = 0;
        v89 = v85 & v19;
        v60 = (__int64 *)(v86 + 16LL * v89);
        v90 = *v60;
        if ( a2 != *v60 )
        {
          while ( v90 != -4096 )
          {
            if ( !v88 && v90 == -8192 )
              v88 = v60;
            v89 = v85 & (v87 + v89);
            v60 = (__int64 *)(v86 + 16LL * v89);
            v90 = *v60;
            if ( a2 == *v60 )
              goto LABEL_107;
            ++v87;
          }
          goto LABEL_112;
        }
LABEL_107:
        v63 = *(_DWORD *)(a1 + 72) + 1;
        goto LABEL_80;
      }
    }
    else
    {
      if ( v7 - *(_DWORD *)(a1 + 76) - v63 > v7 >> 3 )
      {
LABEL_80:
        *(_DWORD *)(a1 + 72) = v63;
        if ( *v60 != -4096 )
          --*(_DWORD *)(a1 + 76);
        *v60 = a2;
        v26 = -2;
        v27 = 0;
        *((_DWORD *)v60 + 2) = 0;
        goto LABEL_11;
      }
      sub_CE2410(v61, v7);
      v91 = *(_DWORD *)(a1 + 80);
      if ( v91 )
      {
        v92 = v91 - 1;
        v93 = *(_QWORD *)(a1 + 64);
        v94 = v92 & v19;
        v60 = (__int64 *)(v93 + 16LL * v94);
        v95 = *v60;
        if ( a2 != *v60 )
        {
          v96 = 1;
          v88 = 0;
          while ( v95 != -4096 )
          {
            if ( v95 == -8192 && !v88 )
              v88 = v60;
            v94 = v92 & (v96 + v94);
            v60 = (__int64 *)(v93 + 16LL * v94);
            v95 = *v60;
            if ( a2 == *v60 )
              goto LABEL_107;
            ++v96;
          }
LABEL_112:
          v63 = *(_DWORD *)(a1 + 72) + 1;
          if ( v88 )
            v60 = v88;
          goto LABEL_80;
        }
        goto LABEL_107;
      }
    }
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
  return (char)v5;
}
