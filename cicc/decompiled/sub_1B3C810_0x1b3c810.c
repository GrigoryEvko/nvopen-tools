// Function: sub_1B3C810
// Address: 0x1b3c810
//
__int64 __fastcall sub_1B3C810(__int64 a1, __int64 a2)
{
  __int64 v2; // r11
  _QWORD *v3; // r10
  __int64 v5; // r12
  unsigned int v6; // esi
  __int64 v7; // r8
  unsigned int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  _QWORD *v11; // rdi
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // rax
  unsigned int v17; // esi
  __int64 v18; // r15
  __int64 v19; // r13
  _QWORD *v20; // r9
  _QWORD *v21; // r8
  unsigned int v22; // edi
  _QWORD *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rax
  unsigned int v28; // r12d
  int v30; // r15d
  _QWORD *v31; // rdx
  int v32; // eax
  int v33; // ecx
  __int64 v34; // rax
  _QWORD *v35; // rdx
  int v36; // eax
  int v37; // eax
  int v38; // esi
  int v39; // esi
  unsigned int v40; // ecx
  __int64 v41; // rdi
  int v42; // r14d
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // rdi
  unsigned int v46; // r14d
  __int64 v47; // rsi
  int v48; // eax
  int v49; // esi
  __int64 v50; // rdi
  unsigned int v51; // eax
  __int64 v52; // r8
  int v53; // r13d
  _QWORD *v54; // r9
  int v55; // eax
  int v56; // eax
  __int64 v57; // rdi
  _QWORD *v58; // r8
  unsigned int v59; // r13d
  int v60; // r9d
  __int64 v61; // rsi
  __int64 *v62; // r14
  _QWORD *v63; // [rsp+0h] [rbp-100h]
  _QWORD *v64; // [rsp+0h] [rbp-100h]
  __int64 *v65; // [rsp+0h] [rbp-100h]
  __int64 v66; // [rsp+0h] [rbp-100h]
  int v67; // [rsp+8h] [rbp-F8h]
  __int64 v68; // [rsp+8h] [rbp-F8h]
  __int64 v69; // [rsp+8h] [rbp-F8h]
  _QWORD *v70; // [rsp+8h] [rbp-F8h]
  __int64 v71; // [rsp+10h] [rbp-F0h]
  __int64 v72; // [rsp+18h] [rbp-E8h]
  __int64 v73; // [rsp+18h] [rbp-E8h]
  __int64 v74; // [rsp+18h] [rbp-E8h]
  _QWORD *v75; // [rsp+20h] [rbp-E0h] BYREF
  __int64 v76; // [rsp+28h] [rbp-D8h]
  _QWORD v77[26]; // [rsp+30h] [rbp-D0h] BYREF

  v2 = a1;
  v3 = v77;
  v77[0] = a2;
  v5 = *(_QWORD *)(a2 + 40);
  v6 = *(_DWORD *)(a1 + 48);
  v76 = 0x1400000001LL;
  v75 = v77;
  v71 = a1 + 24;
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 32);
    v8 = (v6 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v9 = (_QWORD *)(v7 + 16LL * v8);
    v10 = *v9;
    if ( v5 == *v9 )
    {
      v11 = v77;
      *(_QWORD *)(v9[1] + 56LL) = a2;
      v12 = 1;
      goto LABEL_4;
    }
    v30 = 1;
    v31 = 0;
    while ( v10 != -8 )
    {
      if ( v31 || v10 != -16 )
        v9 = v31;
      v8 = (v6 - 1) & (v30 + v8);
      v62 = (__int64 *)(v7 + 16LL * v8);
      v10 = *v62;
      if ( v5 == *v62 )
      {
        v11 = v77;
        *(_QWORD *)(v62[1] + 56) = a2;
        v12 = 1;
        goto LABEL_4;
      }
      ++v30;
      v31 = v9;
      v9 = (_QWORD *)(v7 + 16LL * v8);
    }
    if ( !v31 )
      v31 = v9;
    v32 = *(_DWORD *)(v2 + 40);
    ++*(_QWORD *)(v2 + 24);
    v33 = v32 + 1;
    if ( 4 * (v32 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(v2 + 44) - v33 > v6 >> 3 )
        goto LABEL_23;
      v74 = v2;
      sub_1B3C650(v71, v6);
      v2 = v74;
      v55 = *(_DWORD *)(v74 + 48);
      if ( v55 )
      {
        v56 = v55 - 1;
        v57 = *(_QWORD *)(v74 + 32);
        v58 = 0;
        v59 = v56 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v3 = v77;
        v60 = 1;
        v33 = *(_DWORD *)(v74 + 40) + 1;
        v31 = (_QWORD *)(v57 + 16LL * v59);
        v61 = *v31;
        if ( *v31 != v5 )
        {
          while ( v61 != -8 )
          {
            if ( v61 == -16 && !v58 )
              v58 = v31;
            v59 = v56 & (v60 + v59);
            v31 = (_QWORD *)(v57 + 16LL * v59);
            v61 = *v31;
            if ( v5 == *v31 )
              goto LABEL_23;
            ++v60;
          }
          if ( v58 )
            v31 = v58;
        }
        goto LABEL_23;
      }
LABEL_106:
      ++*(_DWORD *)(v2 + 40);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 24);
  }
  v73 = v2;
  sub_1B3C650(v71, 2 * v6);
  v2 = v73;
  v48 = *(_DWORD *)(v73 + 48);
  if ( !v48 )
    goto LABEL_106;
  v49 = v48 - 1;
  v50 = *(_QWORD *)(v73 + 32);
  v3 = v77;
  v51 = (v48 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v33 = *(_DWORD *)(v73 + 40) + 1;
  v31 = (_QWORD *)(v50 + 16LL * v51);
  v52 = *v31;
  if ( v5 != *v31 )
  {
    v53 = 1;
    v54 = 0;
    while ( v52 != -8 )
    {
      if ( !v54 && v52 == -16 )
        v54 = v31;
      v51 = v49 & (v53 + v51);
      v31 = (_QWORD *)(v50 + 16LL * v51);
      v52 = *v31;
      if ( v5 == *v31 )
        goto LABEL_23;
      ++v53;
    }
    if ( v54 )
      v31 = v54;
  }
LABEL_23:
  *(_DWORD *)(v2 + 40) = v33;
  if ( *v31 != -8 )
    --*(_DWORD *)(v2 + 44);
  *v31 = v5;
  v11 = v75;
  v31[1] = 0;
  v12 = v76;
  MEMORY[0x38] = a2;
LABEL_26:
  while ( 2 )
  {
    if ( v12 )
    {
LABEL_4:
      v13 = v12--;
      v14 = v11[v13 - 1];
      LODWORD(v76) = v12;
      if ( (*(_DWORD *)(v14 + 20) & 0xFFFFFFF) == 0 )
        continue;
      v15 = 0;
      v72 = 8LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
      while ( 1 )
      {
        v16 = (*(_BYTE *)(v14 + 23) & 0x40) != 0
            ? *(_QWORD *)(v14 - 8)
            : v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF);
        v17 = *(_DWORD *)(v2 + 48);
        v18 = *(_QWORD *)(v16 + 3 * v15);
        v19 = *(_QWORD *)(v15 + v16 + 24LL * *(unsigned int *)(v14 + 56) + 8);
        if ( !v17 )
          break;
        LODWORD(v20) = v17 - 1;
        v21 = *(_QWORD **)(v2 + 32);
        v22 = (v17 - 1) & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v23 = &v21[2 * v22];
        v24 = *v23;
        if ( v19 == *v23 )
        {
          v25 = v23[1];
          goto LABEL_11;
        }
        v67 = 1;
        v35 = 0;
        while ( v24 != -8 )
        {
          if ( v35 || v24 != -16 )
            v23 = v35;
          v22 = (unsigned int)v20 & (v67 + v22);
          v65 = &v21[2 * v22];
          v24 = *v65;
          if ( v19 == *v65 )
          {
            v25 = v65[1];
            goto LABEL_11;
          }
          ++v67;
          v35 = v23;
          v23 = &v21[2 * v22];
        }
        if ( !v35 )
          v35 = v23;
        v36 = *(_DWORD *)(v2 + 40);
        ++*(_QWORD *)(v2 + 24);
        v37 = v36 + 1;
        if ( 4 * v37 >= 3 * v17 )
          goto LABEL_47;
        if ( v17 - *(_DWORD *)(v2 + 44) - v37 <= v17 >> 3 )
        {
          v69 = v2;
          v64 = v3;
          sub_1B3C650(v71, v17);
          v2 = v69;
          v43 = *(_DWORD *)(v69 + 48);
          if ( !v43 )
            goto LABEL_106;
          v44 = v43 - 1;
          v45 = *(_QWORD *)(v69 + 32);
          v21 = 0;
          v46 = v44 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
          v3 = v64;
          LODWORD(v20) = 1;
          v37 = *(_DWORD *)(v69 + 40) + 1;
          v35 = (_QWORD *)(v45 + 16LL * v46);
          v47 = *v35;
          if ( *v35 != v19 )
          {
            while ( v47 != -8 )
            {
              if ( !v21 && v47 == -16 )
                v21 = v35;
              v46 = v44 & ((_DWORD)v20 + v46);
              v35 = (_QWORD *)(v45 + 16LL * v46);
              v47 = *v35;
              if ( v19 == *v35 )
                goto LABEL_43;
              LODWORD(v20) = (_DWORD)v20 + 1;
            }
            if ( v21 )
              v35 = v21;
          }
        }
LABEL_43:
        *(_DWORD *)(v2 + 40) = v37;
        if ( *v35 != -8 )
          --*(_DWORD *)(v2 + 44);
        *v35 = v19;
        v25 = 0;
        v35[1] = 0;
LABEL_11:
        v26 = *(_QWORD **)(v25 + 16);
        v27 = v26[1];
        if ( v27 )
          goto LABEL_12;
        if ( *(_BYTE *)(v18 + 16) != 77 || *v26 != *(_QWORD *)(v18 + 40) )
        {
LABEL_13:
          v11 = v75;
          v28 = 0;
          goto LABEL_14;
        }
        v27 = v26[7];
        if ( v27 )
        {
LABEL_12:
          if ( v27 != v18 )
            goto LABEL_13;
        }
        else
        {
          v26[7] = v18;
          v34 = (unsigned int)v76;
          if ( (unsigned int)v76 >= HIDWORD(v76) )
          {
            v66 = v2;
            v70 = v3;
            sub_16CD150((__int64)&v75, v3, 0, 8, (int)v21, (int)v20);
            v34 = (unsigned int)v76;
            v2 = v66;
            v3 = v70;
          }
          v75[v34] = v18;
          LODWORD(v76) = v76 + 1;
        }
        v15 += 8;
        if ( v72 == v15 )
        {
          v12 = v76;
          v11 = v75;
          goto LABEL_26;
        }
      }
      ++*(_QWORD *)(v2 + 24);
LABEL_47:
      v68 = v2;
      v63 = v3;
      sub_1B3C650(v71, 2 * v17);
      v2 = v68;
      v38 = *(_DWORD *)(v68 + 48);
      if ( !v38 )
        goto LABEL_106;
      v39 = v38 - 1;
      v21 = *(_QWORD **)(v68 + 32);
      v3 = v63;
      v40 = v39 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
      v37 = *(_DWORD *)(v68 + 40) + 1;
      v35 = &v21[2 * v40];
      v41 = *v35;
      if ( *v35 != v19 )
      {
        v42 = 1;
        v20 = 0;
        while ( v41 != -8 )
        {
          if ( !v20 && v41 == -16 )
            v20 = v35;
          v40 = v39 & (v42 + v40);
          v35 = &v21[2 * v40];
          v41 = *v35;
          if ( v19 == *v35 )
            goto LABEL_43;
          ++v42;
        }
        if ( v20 )
          v35 = v20;
      }
      goto LABEL_43;
    }
    break;
  }
  v28 = 1;
LABEL_14:
  if ( v11 != v3 )
    _libc_free((unsigned __int64)v11);
  return v28;
}
