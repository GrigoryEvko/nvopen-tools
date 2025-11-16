// Function: sub_1A040C0
// Address: 0x1a040c0
//
__int64 __fastcall sub_1A040C0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // r11
  int v7; // r12d
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // esi
  int v13; // eax
  int v14; // edi
  __int64 v15; // rsi
  unsigned int v16; // edx
  int v17; // ecx
  __int64 v18; // r8
  __int64 result; // rax
  unsigned int v20; // esi
  __int64 v21; // r13
  __int64 v22; // rdi
  unsigned int v23; // ecx
  __int64 *v24; // rax
  __int64 v25; // rdx
  int v26; // r14d
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r12
  unsigned int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // edi
  __int64 *v33; // rax
  __int64 v34; // rcx
  __int64 *v35; // r11
  int v36; // ecx
  int v37; // edi
  int v38; // eax
  int v39; // esi
  __int64 v40; // rcx
  unsigned int v41; // edx
  __int64 v42; // r8
  int v43; // r10d
  __int64 *v44; // r9
  int v45; // eax
  int v46; // ecx
  __int64 v47; // r8
  int v48; // r10d
  unsigned int v49; // edx
  __int64 v50; // rsi
  int v51; // r10d
  __int64 *v52; // r9
  int v53; // edx
  int v54; // ecx
  int v55; // r10d
  __int64 *v56; // r9
  int v57; // edx
  int v58; // eax
  int v59; // edx
  __int64 v60; // rdi
  __int64 *v61; // r8
  unsigned int v62; // r13d
  int v63; // r9d
  __int64 v64; // rsi
  int v65; // eax
  int v66; // edi
  __int64 v67; // rsi
  unsigned int v68; // edx
  __int64 v69; // r8
  int v70; // r10d
  __int64 *v71; // r9
  int v72; // eax
  int v73; // edx
  __int64 v74; // rdi
  __int64 *v75; // r8
  unsigned int v76; // ebx
  int v77; // r9d
  __int64 v78; // rsi
  int v79; // r10d
  __int64 *v80; // r9
  int v81; // [rsp+14h] [rbp-4Ch]
  __int64 v82; // [rsp+18h] [rbp-48h]
  __int64 v83; // [rsp+20h] [rbp-40h]
  int v84; // [rsp+20h] [rbp-40h]
  __int64 v85; // [rsp+20h] [rbp-40h]
  __int64 v87; // [rsp+28h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
  {
    sub_15E08E0(a2, a2);
    v4 = *(_QWORD *)(a2 + 88);
    v5 = v4 + 40LL * *(_QWORD *)(a2 + 96);
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, a2);
      v4 = *(_QWORD *)(a2 + 88);
      if ( v5 != v4 )
        goto LABEL_4;
      goto LABEL_90;
    }
  }
  else
  {
    v4 = *(_QWORD *)(a2 + 88);
    v5 = v4 + 40LL * *(_QWORD *)(a2 + 96);
  }
  if ( v5 != v4 )
  {
LABEL_4:
    v6 = a1 + 32;
    v7 = 2;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 56);
      ++v7;
      if ( !v12 )
        break;
      v8 = *(_QWORD *)(a1 + 40);
      v9 = (v12 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v4 == *v10 )
      {
LABEL_6:
        v4 += 40;
        *((_DWORD *)v10 + 2) = v7;
        if ( v5 == v4 )
          goto LABEL_14;
      }
      else
      {
        v55 = 1;
        v56 = 0;
        while ( v11 != -8 )
        {
          if ( !v56 && v11 == -16 )
            v56 = v10;
          v9 = (v12 - 1) & (v55 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v4 == *v10 )
            goto LABEL_6;
          ++v55;
        }
        v57 = *(_DWORD *)(a1 + 48);
        if ( v56 )
          v10 = v56;
        ++*(_QWORD *)(a1 + 32);
        v17 = v57 + 1;
        if ( 4 * (v57 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 52) - v17 <= v12 >> 3 )
          {
            v85 = v6;
            sub_1A038B0(v6, v12);
            v58 = *(_DWORD *)(a1 + 56);
            if ( !v58 )
            {
LABEL_138:
              ++*(_DWORD *)(a1 + 48);
              BUG();
            }
            v59 = v58 - 1;
            v60 = *(_QWORD *)(a1 + 40);
            v61 = 0;
            v62 = (v58 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
            v6 = v85;
            v63 = 1;
            v17 = *(_DWORD *)(a1 + 48) + 1;
            v10 = (__int64 *)(v60 + 16LL * v62);
            v64 = *v10;
            if ( v4 != *v10 )
            {
              while ( v64 != -8 )
              {
                if ( !v61 && v64 == -16 )
                  v61 = v10;
                v62 = v59 & (v63 + v62);
                v10 = (__int64 *)(v60 + 16LL * v62);
                v64 = *v10;
                if ( v4 == *v10 )
                  goto LABEL_11;
                ++v63;
              }
              if ( v61 )
                v10 = v61;
            }
          }
          goto LABEL_11;
        }
LABEL_9:
        v83 = v6;
        sub_1A038B0(v6, 2 * v12);
        v13 = *(_DWORD *)(a1 + 56);
        if ( !v13 )
          goto LABEL_138;
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 40);
        v6 = v83;
        v16 = (v13 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
        v17 = *(_DWORD *)(a1 + 48) + 1;
        v10 = (__int64 *)(v15 + 16LL * v16);
        v18 = *v10;
        if ( v4 != *v10 )
        {
          v79 = 1;
          v80 = 0;
          while ( v18 != -8 )
          {
            if ( !v80 && v18 == -16 )
              v80 = v10;
            v16 = v14 & (v79 + v16);
            v10 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v10;
            if ( v4 == *v10 )
              goto LABEL_11;
            ++v79;
          }
          if ( v80 )
            v10 = v80;
        }
LABEL_11:
        *(_DWORD *)(a1 + 48) = v17;
        if ( *v10 != -8 )
          --*(_DWORD *)(a1 + 52);
        *v10 = v4;
        v4 += 40;
        *((_DWORD *)v10 + 2) = 0;
        *((_DWORD *)v10 + 2) = v7;
        if ( v5 == v4 )
          goto LABEL_14;
      }
    }
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_9;
  }
LABEL_90:
  v7 = 2;
LABEL_14:
  result = *a3;
  v84 = (v7 + 1) << 16;
  v87 = a3[1];
  v82 = result;
  while ( v82 != v87 )
  {
    v20 = *(_DWORD *)(a1 + 24);
    v21 = *(_QWORD *)(v87 - 8);
    if ( v20 )
    {
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
      v24 = (__int64 *)(v22 + 16LL * v23);
      v25 = *v24;
      if ( v21 == *v24 )
        goto LABEL_17;
      v51 = 1;
      v52 = 0;
      while ( v25 != -8 )
      {
        if ( v25 == -16 && !v52 )
          v52 = v24;
        v23 = (v20 - 1) & (v51 + v23);
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v21 == *v24 )
          goto LABEL_17;
        ++v51;
      }
      v53 = *(_DWORD *)(a1 + 16);
      if ( v52 )
        v24 = v52;
      ++*(_QWORD *)a1;
      v54 = v53 + 1;
      if ( 4 * (v53 + 1) < 3 * v20 )
      {
        if ( v20 - *(_DWORD *)(a1 + 20) - v54 <= v20 >> 3 )
        {
          sub_13FEAC0(a1, v20);
          v72 = *(_DWORD *)(a1 + 24);
          if ( !v72 )
          {
LABEL_136:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v73 = v72 - 1;
          v74 = *(_QWORD *)(a1 + 8);
          v75 = 0;
          v76 = (v72 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v77 = 1;
          v54 = *(_DWORD *)(a1 + 16) + 1;
          v24 = (__int64 *)(v74 + 16LL * v76);
          v78 = *v24;
          if ( v21 != *v24 )
          {
            while ( v78 != -8 )
            {
              if ( !v75 && v78 == -16 )
                v75 = v24;
              v76 = v73 & (v77 + v76);
              v24 = (__int64 *)(v74 + 16LL * v76);
              v78 = *v24;
              if ( v21 == *v24 )
                goto LABEL_59;
              ++v77;
            }
            if ( v75 )
              v24 = v75;
          }
        }
        goto LABEL_59;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_13FEAC0(a1, 2 * v20);
    v65 = *(_DWORD *)(a1 + 24);
    if ( !v65 )
      goto LABEL_136;
    v66 = v65 - 1;
    v67 = *(_QWORD *)(a1 + 8);
    v68 = (v65 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
    v54 = *(_DWORD *)(a1 + 16) + 1;
    v24 = (__int64 *)(v67 + 16LL * v68);
    v69 = *v24;
    if ( v21 != *v24 )
    {
      v70 = 1;
      v71 = 0;
      while ( v69 != -8 )
      {
        if ( v69 == -16 && !v71 )
          v71 = v24;
        v68 = v66 & (v70 + v68);
        v24 = (__int64 *)(v67 + 16LL * v68);
        v69 = *v24;
        if ( v21 == *v24 )
          goto LABEL_59;
        ++v70;
      }
      if ( v71 )
        v24 = v71;
    }
LABEL_59:
    *(_DWORD *)(a1 + 16) = v54;
    if ( *v24 != -8 )
      --*(_DWORD *)(a1 + 20);
    *v24 = v21;
    *((_DWORD *)v24 + 2) = 0;
LABEL_17:
    v26 = v84;
    v27 = v21 + 40;
    *((_DWORD *)v24 + 2) = v84;
    v28 = *(_QWORD *)(v27 + 8);
    if ( v27 != v28 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
          v29 = v28 - 24;
          if ( !v28 )
            v29 = 0;
          if ( (unsigned __int8)sub_14AF890(v29) )
            break;
          v28 = *(_QWORD *)(v28 + 8);
          if ( v27 == v28 )
            goto LABEL_26;
        }
        v30 = *(_DWORD *)(a1 + 56);
        ++v26;
        if ( !v30 )
          break;
        v31 = *(_QWORD *)(a1 + 40);
        v32 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v33 = (__int64 *)(v31 + 16LL * v32);
        v34 = *v33;
        if ( v29 != *v33 )
        {
          v81 = 1;
          v35 = 0;
          while ( v34 != -8 )
          {
            if ( v34 == -16 && !v35 )
              v35 = v33;
            v32 = (v30 - 1) & (v81 + v32);
            v33 = (__int64 *)(v31 + 16LL * v32);
            v34 = *v33;
            if ( v29 == *v33 )
              goto LABEL_25;
            ++v81;
          }
          v36 = *(_DWORD *)(a1 + 48);
          if ( v35 )
            v33 = v35;
          ++*(_QWORD *)(a1 + 32);
          v37 = v36 + 1;
          if ( 4 * (v36 + 1) < 3 * v30 )
          {
            if ( v30 - *(_DWORD *)(a1 + 52) - v37 <= v30 >> 3 )
            {
              sub_1A038B0(a1 + 32, v30);
              v45 = *(_DWORD *)(a1 + 56);
              if ( !v45 )
              {
LABEL_137:
                ++*(_DWORD *)(a1 + 48);
                BUG();
              }
              v46 = v45 - 1;
              v47 = *(_QWORD *)(a1 + 40);
              v48 = 1;
              v44 = 0;
              v49 = (v45 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
              v37 = *(_DWORD *)(a1 + 48) + 1;
              v33 = (__int64 *)(v47 + 16LL * v49);
              v50 = *v33;
              if ( v29 != *v33 )
              {
                while ( v50 != -8 )
                {
                  if ( !v44 && v50 == -16 )
                    v44 = v33;
                  v49 = v46 & (v48 + v49);
                  v33 = (__int64 *)(v47 + 16LL * v49);
                  v50 = *v33;
                  if ( v29 == *v33 )
                    goto LABEL_34;
                  ++v48;
                }
                goto LABEL_50;
              }
            }
            goto LABEL_34;
          }
LABEL_38:
          sub_1A038B0(a1 + 32, 2 * v30);
          v38 = *(_DWORD *)(a1 + 56);
          if ( !v38 )
            goto LABEL_137;
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a1 + 40);
          v41 = (v38 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
          v37 = *(_DWORD *)(a1 + 48) + 1;
          v33 = (__int64 *)(v40 + 16LL * v41);
          v42 = *v33;
          if ( v29 != *v33 )
          {
            v43 = 1;
            v44 = 0;
            while ( v42 != -8 )
            {
              if ( v42 == -16 && !v44 )
                v44 = v33;
              v41 = v39 & (v43 + v41);
              v33 = (__int64 *)(v40 + 16LL * v41);
              v42 = *v33;
              if ( v29 == *v33 )
                goto LABEL_34;
              ++v43;
            }
LABEL_50:
            if ( v44 )
              v33 = v44;
          }
LABEL_34:
          *(_DWORD *)(a1 + 48) = v37;
          if ( *v33 != -8 )
            --*(_DWORD *)(a1 + 52);
          *v33 = v29;
          *((_DWORD *)v33 + 2) = 0;
        }
LABEL_25:
        *((_DWORD *)v33 + 2) = v26;
        v28 = *(_QWORD *)(v28 + 8);
        if ( v27 == v28 )
          goto LABEL_26;
      }
      ++*(_QWORD *)(a1 + 32);
      goto LABEL_38;
    }
LABEL_26:
    v87 -= 8;
    result = v87;
    v84 += 0x10000;
  }
  return result;
}
