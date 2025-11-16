// Function: sub_1DD0CF0
// Address: 0x1dd0cf0
//
void __fastcall sub_1DD0CF0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned __int16 *v5; // rbx
  __int64 v6; // r8
  int v7; // r9d
  unsigned __int16 *i; // r15
  unsigned int v9; // esi
  int v10; // eax
  __int64 v11; // rdx
  _QWORD *v12; // rax
  _QWORD *k; // rdx
  __int64 v14; // rbx
  __int64 v15; // r14
  int v16; // r11d
  unsigned int v17; // esi
  int v18; // r13d
  __int64 v19; // r9
  __int64 v20; // r8
  unsigned int v21; // edi
  __int64 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned int *v26; // r14
  __int64 v27; // rbx
  __int64 v28; // rsi
  __int64 v29; // r15
  __int64 *v30; // rax
  __int64 *v31; // r15
  __int64 v32; // r14
  __int64 *v33; // rax
  __int64 v34; // rdi
  unsigned __int16 *v35; // r12
  unsigned __int16 *m; // rbx
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rbx
  _DWORD *v40; // rax
  _DWORD *v41; // rdx
  __int64 v42; // rax
  int *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  unsigned int v46; // ecx
  _QWORD *v47; // rdi
  unsigned int v48; // eax
  int v49; // eax
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rax
  int v52; // ebx
  __int64 v53; // r14
  _QWORD *v54; // rax
  __int64 v55; // rdx
  _QWORD *j; // rdx
  int v57; // r10d
  __int64 v58; // rdx
  int v59; // eax
  int v60; // eax
  int v61; // r15d
  int v62; // r15d
  __int64 v63; // r10
  int v64; // edi
  __int64 v65; // rsi
  int v66; // r10d
  int v67; // r10d
  int v68; // esi
  unsigned int v69; // r15d
  __int64 v70; // rdi
  _QWORD *v71; // rax
  int v72; // [rsp+0h] [rbp-E0h]
  int v73; // [rsp+0h] [rbp-E0h]
  __int64 v74; // [rsp+8h] [rbp-D8h]
  __int64 v75; // [rsp+10h] [rbp-D0h]
  __int64 *v77; // [rsp+28h] [rbp-B8h]
  unsigned int v78; // [rsp+3Ch] [rbp-A4h] BYREF
  unsigned __int64 v79[2]; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE v80[16]; // [rsp+50h] [rbp-90h] BYREF
  _BYTE *v81; // [rsp+60h] [rbp-80h] BYREF
  __int64 v82; // [rsp+68h] [rbp-78h]
  _BYTE v83[24]; // [rsp+70h] [rbp-70h] BYREF
  int v84; // [rsp+88h] [rbp-58h] BYREF
  __int64 v85; // [rsp+90h] [rbp-50h]
  int *v86; // [rsp+98h] [rbp-48h]
  int *v87; // [rsp+A0h] [rbp-40h]
  __int64 v88; // [rsp+A8h] [rbp-38h]

  v3 = a2;
  v4 = a1;
  v5 = *(unsigned __int16 **)(a2 + 160);
  v79[0] = (unsigned __int64)v80;
  v79[1] = 0x400000000LL;
  for ( i = (unsigned __int16 *)sub_1DD77D0(a2); v5 != i; i += 4 )
  {
    v9 = *i;
    sub_1DCFC50((_QWORD *)a1, v9, 0, (__int64)v79, v6, v7);
  }
  ++*(_QWORD *)(a1 + 440);
  v74 = a1 + 440;
  v10 = *(_DWORD *)(a1 + 456);
  if ( v10 )
  {
    v46 = 4 * v10;
    v11 = *(unsigned int *)(a1 + 464);
    if ( (unsigned int)(4 * v10) < 0x40 )
      v46 = 64;
    if ( v46 >= (unsigned int)v11 )
      goto LABEL_6;
    v47 = *(_QWORD **)(a1 + 448);
    v48 = v10 - 1;
    if ( v48 )
    {
      _BitScanReverse(&v48, v48);
      v49 = 1 << (33 - (v48 ^ 0x1F));
      if ( v49 < 64 )
        v49 = 64;
      if ( (_DWORD)v11 == v49 )
      {
        *(_QWORD *)(v4 + 456) = 0;
        v71 = &v47[2 * (unsigned int)v11];
        do
        {
          if ( v47 )
            *v47 = -8;
          v47 += 2;
        }
        while ( v71 != v47 );
        goto LABEL_9;
      }
      v50 = (4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1);
      v51 = ((v50 | (v50 >> 2)) >> 4) | v50 | (v50 >> 2) | ((((v50 | (v50 >> 2)) >> 4) | v50 | (v50 >> 2)) >> 8);
      v52 = (v51 | (v51 >> 16)) + 1;
      v53 = 16 * ((v51 | (v51 >> 16)) + 1);
    }
    else
    {
      v53 = 2048;
      v52 = 128;
    }
    j___libc_free_0(v47);
    *(_DWORD *)(v4 + 464) = v52;
    v54 = (_QWORD *)sub_22077B0(v53);
    v55 = *(unsigned int *)(v4 + 464);
    *(_QWORD *)(v4 + 456) = 0;
    *(_QWORD *)(v4 + 448) = v54;
    for ( j = &v54[2 * v55]; j != v54; v54 += 2 )
    {
      if ( v54 )
        *v54 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 460) )
  {
    v11 = *(unsigned int *)(a1 + 464);
    if ( (unsigned int)v11 <= 0x40 )
    {
LABEL_6:
      v12 = *(_QWORD **)(a1 + 448);
      for ( k = &v12[2 * v11]; k != v12; v12 += 2 )
        *v12 = -8;
      *(_QWORD *)(a1 + 456) = 0;
      goto LABEL_9;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 448));
    *(_QWORD *)(a1 + 448) = 0;
    *(_QWORD *)(a1 + 456) = 0;
    *(_DWORD *)(a1 + 464) = 0;
  }
LABEL_9:
  v14 = *(_QWORD *)(v3 + 32);
  v15 = v3 + 24;
  v16 = 0;
  if ( v14 == v3 + 24 )
    goto LABEL_21;
  v75 = v3;
  do
  {
    while ( 1 )
    {
      if ( (unsigned __int16)(**(_WORD **)(v14 + 16) - 12) <= 1u )
        goto LABEL_11;
      v17 = *(_DWORD *)(v4 + 464);
      v18 = v16 + 1;
      if ( !v17 )
      {
        ++*(_QWORD *)(v4 + 440);
        goto LABEL_82;
      }
      LODWORD(v19) = v17 - 1;
      v20 = *(_QWORD *)(v4 + 448);
      v21 = (v17 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v22 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v22;
      if ( *v22 != v14 )
      {
        v57 = 1;
        v58 = 0;
        while ( v23 != -8 )
        {
          if ( v58 || v23 != -16 )
            v22 = (__int64 *)v58;
          v21 = v19 & (v57 + v21);
          v23 = *(_QWORD *)(v20 + 16LL * v21);
          if ( v23 == v14 )
            goto LABEL_16;
          ++v57;
          v58 = (__int64)v22;
          v22 = (__int64 *)(v20 + 16LL * v21);
        }
        if ( !v58 )
          v58 = (__int64)v22;
        v59 = *(_DWORD *)(v4 + 456);
        ++*(_QWORD *)(v4 + 440);
        v60 = v59 + 1;
        if ( 4 * v60 < 3 * v17 )
        {
          v23 = v17 - *(_DWORD *)(v4 + 460) - v60;
          if ( (unsigned int)v23 <= v17 >> 3 )
          {
            v73 = v16;
            sub_1DC6D40(v74, v17);
            v66 = *(_DWORD *)(v4 + 464);
            if ( !v66 )
            {
LABEL_116:
              ++*(_DWORD *)(v4 + 456);
              BUG();
            }
            v67 = v66 - 1;
            v23 = 0;
            v16 = v73;
            v68 = 1;
            v69 = v67 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = *(_QWORD *)(v4 + 448);
            v60 = *(_DWORD *)(v4 + 456) + 1;
            v58 = v19 + 16LL * v69;
            v70 = *(_QWORD *)v58;
            if ( v14 != *(_QWORD *)v58 )
            {
              while ( v70 != -8 )
              {
                if ( v70 == -16 && !v23 )
                  v23 = v58;
                v20 = (unsigned int)(v68 + 1);
                v69 = v67 & (v68 + v69);
                v58 = v19 + 16LL * v69;
                v70 = *(_QWORD *)v58;
                if ( *(_QWORD *)v58 == v14 )
                  goto LABEL_77;
                ++v68;
              }
              if ( v23 )
                v58 = v23;
            }
          }
          goto LABEL_77;
        }
LABEL_82:
        v72 = v16;
        sub_1DC6D40(v74, 2 * v17);
        v61 = *(_DWORD *)(v4 + 464);
        if ( !v61 )
          goto LABEL_116;
        v62 = v61 - 1;
        v63 = *(_QWORD *)(v4 + 448);
        v16 = v72;
        v23 = v62 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v60 = *(_DWORD *)(v4 + 456) + 1;
        v58 = v63 + 16 * v23;
        v20 = *(_QWORD *)v58;
        if ( *(_QWORD *)v58 != v14 )
        {
          v64 = 1;
          v65 = 0;
          while ( v20 != -8 )
          {
            if ( v20 == -16 && !v65 )
              v65 = v58;
            LODWORD(v19) = v64 + 1;
            v23 = v62 & (unsigned int)(v64 + v23);
            v58 = v63 + 16LL * (unsigned int)v23;
            v20 = *(_QWORD *)v58;
            if ( *(_QWORD *)v58 == v14 )
              goto LABEL_77;
            ++v64;
          }
          if ( v65 )
            v58 = v65;
        }
LABEL_77:
        *(_DWORD *)(v4 + 456) = v60;
        if ( *(_QWORD *)v58 != -8 )
          --*(_DWORD *)(v4 + 460);
        *(_QWORD *)v58 = v14;
        *(_DWORD *)(v58 + 8) = v16;
      }
LABEL_16:
      sub_1DD0830((char *)v4, v14, (__int64)v79, v23, v20, v19);
      v16 = v18;
LABEL_11:
      if ( (*(_BYTE *)v14 & 4) == 0 )
        break;
      v14 = *(_QWORD *)(v14 + 8);
      if ( v15 == v14 )
        goto LABEL_20;
    }
    while ( (*(_BYTE *)(v14 + 46) & 8) != 0 )
      v14 = *(_QWORD *)(v14 + 8);
    v14 = *(_QWORD *)(v14 + 8);
  }
  while ( v15 != v14 );
LABEL_20:
  v3 = v75;
LABEL_21:
  v24 = *(_QWORD *)(v4 + 416) + 32LL * *(int *)(v3 + 48);
  v25 = *(unsigned int *)(v24 + 8);
  if ( (_DWORD)v25 )
  {
    v26 = *(unsigned int **)v24;
    v27 = *(_QWORD *)v24 + 4 * v25;
    do
    {
      v28 = *v26++;
      v29 = *(_QWORD *)(sub_1E69D00(*(_QWORD *)(v4 + 352), v28) + 24);
      v30 = (__int64 *)sub_1DCC790((char *)v4, *(v26 - 1));
      sub_1DCBEC0(v4, v30, v29, v3);
    }
    while ( v26 != (unsigned int *)v27 );
  }
  v31 = *(__int64 **)(v3 + 88);
  v84 = 0;
  v32 = v4;
  v81 = v83;
  v82 = 0x400000000LL;
  v86 = &v84;
  v87 = &v84;
  v33 = *(__int64 **)(v3 + 96);
  v85 = 0;
  v88 = 0;
  v77 = v33;
  if ( v33 != v31 )
  {
    do
    {
      v34 = *v31;
      if ( !*(_BYTE *)(*v31 + 180) )
      {
        v35 = *(unsigned __int16 **)(v34 + 160);
        for ( m = (unsigned __int16 *)sub_1DD77D0(v34); v35 != m; m += 4 )
        {
          v37 = *m;
          v38 = *(_QWORD *)(*(_QWORD *)(v32 + 360) + 232LL);
          if ( !*(_BYTE *)(v38 + 8 * v37 + 4) )
          {
            v78 = *m;
            sub_1D041C0((__int64)&v81, &v78, v38, v37, v6);
          }
        }
      }
      ++v31;
    }
    while ( v77 != v31 );
    v4 = v32;
  }
  v39 = 0;
  if ( a3 )
  {
    while ( 2 )
    {
      if ( !*(_QWORD *)(*(_QWORD *)(v4 + 368) + 8 * v39) && !*(_QWORD *)(*(_QWORD *)(v4 + 392) + 8 * v39) )
        goto LABEL_42;
      if ( v88 )
      {
        v42 = v85;
        if ( v85 )
        {
          v43 = &v84;
          do
          {
            while ( 1 )
            {
              v44 = *(_QWORD *)(v42 + 16);
              v45 = *(_QWORD *)(v42 + 24);
              if ( *(_DWORD *)(v42 + 32) >= (unsigned int)v39 )
                break;
              v42 = *(_QWORD *)(v42 + 24);
              if ( !v45 )
                goto LABEL_53;
            }
            v43 = (int *)v42;
            v42 = *(_QWORD *)(v42 + 16);
          }
          while ( v44 );
LABEL_53:
          if ( v43 != &v84 && v43[8] <= (unsigned int)v39 )
          {
LABEL_42:
            if ( ++v39 == a3 )
              goto LABEL_43;
            continue;
          }
        }
      }
      else
      {
        v40 = v81;
        v41 = &v81[4 * (unsigned int)v82];
        if ( v81 != (_BYTE *)v41 )
        {
          while ( *v40 != (_DWORD)v39 )
          {
            if ( v41 == ++v40 )
              goto LABEL_55;
          }
          if ( v41 != v40 )
            goto LABEL_42;
        }
      }
      break;
    }
LABEL_55:
    sub_1DCFC50((_QWORD *)v4, v39, 0, (__int64)v79, v6, v7);
    goto LABEL_42;
  }
LABEL_43:
  sub_1DCADB0(v85);
  if ( v81 != v83 )
    _libc_free((unsigned __int64)v81);
  if ( (_BYTE *)v79[0] != v80 )
    _libc_free(v79[0]);
}
