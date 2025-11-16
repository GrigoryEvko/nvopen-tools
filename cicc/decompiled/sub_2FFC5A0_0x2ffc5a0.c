// Function: sub_2FFC5A0
// Address: 0x2ffc5a0
//
__int64 __fastcall sub_2FFC5A0(__int64 a1, __int64 *a2, _QWORD *a3, int a4, int a5, _DWORD *a6)
{
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rsi
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // r15
  __int64 i; // r15
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v16; // r14
  _DWORD *v17; // r11
  __int64 v18; // r13
  unsigned int v19; // esi
  __int64 v20; // rbx
  __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rbx
  _QWORD *v24; // r12
  _QWORD *v25; // r13
  _QWORD *v26; // r14
  unsigned __int64 *v27; // rdi
  unsigned __int64 v28; // rdx
  __int64 v29; // rbx
  __int64 v30; // r9
  _DWORD *v31; // r15
  int v32; // r14d
  unsigned int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // edi
  _QWORD *v36; // rax
  __int64 v37; // rcx
  int v38; // eax
  __int64 v39; // rdi
  int v40; // edx
  unsigned int v41; // eax
  int *v42; // rcx
  int v43; // esi
  int v44; // eax
  __int64 v45; // rdi
  int v46; // eax
  unsigned int v47; // edx
  int *v48; // rcx
  int v49; // esi
  int v50; // r11d
  _QWORD *v51; // rdx
  int v52; // eax
  int v53; // eax
  __int64 v54; // r13
  __int64 v55; // rbx
  unsigned int v56; // eax
  __int64 v57; // r8
  __int64 v58; // r9
  int v59; // esi
  int v60; // esi
  __int64 v61; // r10
  unsigned int v62; // ecx
  __int64 v63; // rdi
  int v64; // r13d
  _QWORD *v65; // r8
  int v66; // r11d
  int v67; // r11d
  __int64 v68; // r8
  _QWORD *v69; // rcx
  unsigned int v70; // r13d
  int v71; // esi
  __int64 v72; // rdi
  int v73; // ecx
  int v74; // r8d
  int v75; // ecx
  int v76; // r8d
  _DWORD *v77; // [rsp+0h] [rbp-70h]
  __int64 v78; // [rsp+8h] [rbp-68h]
  __int64 v80; // [rsp+10h] [rbp-60h]
  __int64 v81; // [rsp+10h] [rbp-60h]
  __int64 v85; // [rsp+28h] [rbp-48h]
  __int64 v86; // [rsp+28h] [rbp-48h]
  __int64 v87; // [rsp+30h] [rbp-40h]
  __int64 v88; // [rsp+30h] [rbp-40h]
  __int64 v89; // [rsp+30h] [rbp-40h]

  v6 = a1;
  v7 = *(_QWORD *)(a1 + 72);
  v8 = *a2;
  v85 = v7;
  if ( *(_QWORD *)(v7 + 56) == v8 )
  {
    v10 = v7 + 48;
    if ( !v8 )
      BUG();
    v12 = v8;
    if ( (*(_QWORD *)v8 & 4) == 0 )
      goto LABEL_9;
  }
  else
  {
    v9 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v9 )
      BUG();
    v10 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v9 & 4) == 0 && (*(_BYTE *)(v9 + 44) & 4) != 0 )
    {
      for ( i = *(_QWORD *)v9; ; i = *(_QWORD *)v10 )
      {
        v10 = i & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)(v10 + 44) & 4) == 0 )
          break;
      }
    }
    v12 = v8;
    if ( (*(_QWORD *)v8 & 4) == 0 )
    {
LABEL_9:
      if ( (*(_BYTE *)(v8 + 44) & 8) != 0 )
      {
        do
          v12 = *(_QWORD *)(v12 + 8);
        while ( (*(_BYTE *)(v12 + 44) & 8) != 0 );
      }
    }
  }
  v13 = *(_QWORD *)(a1 + 8);
  v87 = *(_QWORD *)(v12 + 8);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 256LL);
  if ( v14 == sub_2FDC490 )
    return 0;
  v16 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v14)(
          v13,
          v8,
          *(_QWORD *)(v6 + 40),
          *(_QWORD *)(v6 + 48));
  if ( !v16 )
    return 0;
  v17 = a6;
  v18 = *a2;
  v19 = *(_DWORD *)(*a2 + 64);
  if ( v19 )
  {
    sub_2E88FE0(*a2);
    v54 = (unsigned int)sub_2EAB0A0(*(_QWORD *)(v18 + 32));
    sub_2E88FE0(v16);
    v55 = (unsigned int)sub_2EAB0A0(*(_QWORD *)(v16 + 32));
    v56 = sub_2E8E690(v16);
    sub_2E79810(*(_QWORD *)v6, (v54 << 32) | v19, (v55 << 32) | v56, 0, v57, v58);
    v20 = *(_QWORD *)(v6 + 72);
    v17 = a6;
    v18 = *a2;
    if ( !*a2 )
      BUG();
  }
  else
  {
    v20 = *(_QWORD *)(v6 + 72);
  }
  v21 = v18;
  if ( (*(_BYTE *)v18 & 4) == 0 && (*(_BYTE *)(v18 + 44) & 8) != 0 )
  {
    do
      v21 = *(_QWORD *)(v21 + 8);
    while ( (*(_BYTE *)(v21 + 44) & 8) != 0 );
  }
  v22 = *(_QWORD **)(v21 + 8);
  if ( (_QWORD *)v18 != v22 )
  {
    v80 = v16;
    v23 = v20 + 40;
    v77 = v17;
    v78 = v6;
    v24 = (_QWORD *)v18;
    v25 = v22;
    do
    {
      v26 = v24;
      v24 = (_QWORD *)v24[1];
      sub_2E31080(v23, (__int64)v26);
      v27 = (unsigned __int64 *)v26[1];
      v28 = *v26 & 0xFFFFFFFFFFFFFFF8LL;
      *v27 = v28 | *v27 & 7;
      *(_QWORD *)(v28 + 8) = v27;
      *v26 &= 7uLL;
      v26[1] = 0;
      sub_2E310F0(v23);
    }
    while ( v24 != v25 );
    v16 = v80;
    v6 = v78;
    v17 = v77;
  }
  if ( v10 == v85 + 48 )
  {
    v29 = *(_QWORD *)(v85 + 56);
  }
  else
  {
    if ( (*(_BYTE *)v10 & 4) == 0 && (*(_BYTE *)(v10 + 44) & 8) != 0 )
    {
      do
        v10 = *(_QWORD *)(v10 + 8);
      while ( (*(_BYTE *)(v10 + 44) & 8) != 0 );
    }
    v29 = *(_QWORD *)(v10 + 8);
  }
  v81 = v6 + 80;
  if ( v29 != v87 )
  {
    v86 = v16;
    v30 = v87;
    v31 = v17;
    while ( 1 )
    {
      v32 = (*v31)++;
      v33 = *(_DWORD *)(v6 + 104);
      if ( !v33 )
        break;
      v34 = *(_QWORD *)(v6 + 88);
      v35 = (v33 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
      v36 = (_QWORD *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( v29 != *v36 )
      {
        v50 = 1;
        v51 = 0;
        while ( v37 != -4096 )
        {
          if ( v51 || v37 != -8192 )
            v36 = v51;
          v35 = (v33 - 1) & (v50 + v35);
          v37 = *(_QWORD *)(v34 + 16LL * v35);
          if ( v37 == v29 )
            goto LABEL_31;
          ++v50;
          v51 = v36;
          v36 = (_QWORD *)(v34 + 16LL * v35);
        }
        if ( !v51 )
          v51 = v36;
        v52 = *(_DWORD *)(v6 + 96);
        ++*(_QWORD *)(v6 + 80);
        v53 = v52 + 1;
        if ( 4 * v53 < 3 * v33 )
        {
          if ( v33 - *(_DWORD *)(v6 + 100) - v53 <= v33 >> 3 )
          {
            v89 = v30;
            sub_2E261E0(v81, v33);
            v66 = *(_DWORD *)(v6 + 104);
            if ( !v66 )
            {
LABEL_106:
              ++*(_DWORD *)(v6 + 96);
              BUG();
            }
            v67 = v66 - 1;
            v68 = *(_QWORD *)(v6 + 88);
            v69 = 0;
            v70 = v67 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
            v30 = v89;
            v71 = 1;
            v53 = *(_DWORD *)(v6 + 96) + 1;
            v51 = (_QWORD *)(v68 + 16LL * v70);
            v72 = *v51;
            if ( *v51 != v29 )
            {
              while ( v72 != -4096 )
              {
                if ( !v69 && v72 == -8192 )
                  v69 = v51;
                v70 = v67 & (v71 + v70);
                v51 = (_QWORD *)(v68 + 16LL * v70);
                v72 = *v51;
                if ( *v51 == v29 )
                  goto LABEL_58;
                ++v71;
              }
              if ( v69 )
                v51 = v69;
            }
          }
          goto LABEL_58;
        }
LABEL_65:
        v88 = v30;
        sub_2E261E0(v81, 2 * v33);
        v59 = *(_DWORD *)(v6 + 104);
        if ( !v59 )
          goto LABEL_106;
        v60 = v59 - 1;
        v61 = *(_QWORD *)(v6 + 88);
        v30 = v88;
        v62 = v60 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
        v53 = *(_DWORD *)(v6 + 96) + 1;
        v51 = (_QWORD *)(v61 + 16LL * v62);
        v63 = *v51;
        if ( *v51 != v29 )
        {
          v64 = 1;
          v65 = 0;
          while ( v63 != -4096 )
          {
            if ( !v65 && v63 == -8192 )
              v65 = v51;
            v62 = v60 & (v64 + v62);
            v51 = (_QWORD *)(v61 + 16LL * v62);
            v63 = *v51;
            if ( *v51 == v29 )
              goto LABEL_58;
            ++v64;
          }
          if ( v65 )
            v51 = v65;
        }
LABEL_58:
        *(_DWORD *)(v6 + 96) = v53;
        if ( *v51 != -4096 )
          --*(_DWORD *)(v6 + 100);
        *v51 = v29;
        *((_DWORD *)v51 + 2) = v32;
      }
LABEL_31:
      if ( !v29 )
        BUG();
      if ( (*(_BYTE *)v29 & 4) != 0 )
      {
        v29 = *(_QWORD *)(v29 + 8);
        if ( v29 == v30 )
          goto LABEL_34;
      }
      else
      {
        while ( (*(_BYTE *)(v29 + 44) & 8) != 0 )
          v29 = *(_QWORD *)(v29 + 8);
        v29 = *(_QWORD *)(v29 + 8);
        if ( v29 == v30 )
        {
LABEL_34:
          v16 = v86;
          v17 = v31;
          goto LABEL_35;
        }
      }
    }
    ++*(_QWORD *)(v6 + 80);
    goto LABEL_65;
  }
LABEL_35:
  --*v17;
  *a2 = v16;
  if ( (*(_BYTE *)v16 & 4) == 0 )
  {
    while ( (*(_BYTE *)(v16 + 44) & 8) != 0 )
      v16 = *(_QWORD *)(v16 + 8);
  }
  *a3 = *(_QWORD *)(v16 + 8);
  v38 = *(_DWORD *)(v6 + 232);
  v39 = *(_QWORD *)(v6 + 216);
  if ( v38 )
  {
    v40 = v38 - 1;
    v41 = (v38 - 1) & (37 * a4);
    v42 = (int *)(v39 + 8LL * v41);
    v43 = *v42;
    if ( a4 == *v42 )
    {
LABEL_38:
      *v42 = -2;
      --*(_DWORD *)(v6 + 224);
      ++*(_DWORD *)(v6 + 228);
    }
    else
    {
      v75 = 1;
      while ( v43 != -1 )
      {
        v76 = v75 + 1;
        v41 = v40 & (v75 + v41);
        v42 = (int *)(v39 + 8LL * v41);
        v43 = *v42;
        if ( a4 == *v42 )
          goto LABEL_38;
        v75 = v76;
      }
    }
  }
  v44 = *(_DWORD *)(v6 + 264);
  v45 = *(_QWORD *)(v6 + 248);
  if ( v44 )
  {
    v46 = v44 - 1;
    v47 = v46 & (37 * a5);
    v48 = (int *)(v45 + 8LL * v47);
    v49 = *v48;
    if ( a5 == *v48 )
    {
LABEL_41:
      *v48 = -2;
      --*(_DWORD *)(v6 + 256);
      ++*(_DWORD *)(v6 + 260);
    }
    else
    {
      v73 = 1;
      while ( v49 != -1 )
      {
        v74 = v73 + 1;
        v47 = v46 & (v73 + v47);
        v48 = (int *)(v45 + 8LL * v47);
        v49 = *v48;
        if ( a5 == *v48 )
          goto LABEL_41;
        v73 = v74;
      }
    }
  }
  return 1;
}
