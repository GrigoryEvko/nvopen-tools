// Function: sub_1F112A0
// Address: 0x1f112a0
//
__int64 __fastcall sub_1F112A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v6; // r11
  __int64 v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // edx
  unsigned __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r11
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // r11d
  __int64 v22; // r13
  __int64 v23; // rbx
  int v24; // r12d
  unsigned __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r14
  unsigned __int64 v29; // r12
  __int64 v30; // rax
  __int64 v31; // rbx
  unsigned __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 *v35; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // r14
  __int64 v40; // rax
  int v41; // r8d
  int v42; // r9d
  __int64 v43; // rdx
  unsigned __int64 v44; // r8
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // rax
  __int64 v49; // r11
  unsigned int v50; // edi
  _QWORD *v51; // rax
  __int64 v52; // rcx
  _QWORD *v53; // rdx
  int v54; // eax
  int v55; // eax
  int v56; // r9d
  __int64 v57; // r10
  unsigned int v58; // ecx
  int v59; // edi
  _QWORD *v60; // rsi
  __int64 *v61; // r13
  __int64 v62; // rax
  unsigned __int64 v63; // rbx
  __int64 *v64; // r12
  unsigned __int64 v65; // rsi
  __int64 *v66; // rbx
  __int64 *v67; // rdi
  int v69; // r8d
  _QWORD *v70; // rcx
  int v71; // esi
  unsigned int v72; // r14d
  __int64 v73; // rdi
  _QWORD *v74; // rax
  __int64 i; // rdx
  __int64 v76; // [rsp+8h] [rbp-58h]
  unsigned __int64 v77; // [rsp+10h] [rbp-50h]
  int v78; // [rsp+18h] [rbp-48h]
  unsigned int v79; // [rsp+18h] [rbp-48h]
  int v80; // [rsp+18h] [rbp-48h]
  __int64 v81; // [rsp+18h] [rbp-48h]
  __int64 v82; // [rsp+18h] [rbp-48h]
  int v83; // [rsp+18h] [rbp-48h]
  __int64 v84; // [rsp+18h] [rbp-48h]
  int v85; // [rsp+18h] [rbp-48h]
  unsigned __int64 v86; // [rsp+20h] [rbp-40h]
  __int64 v87; // [rsp+28h] [rbp-38h]
  __int64 v88; // [rsp+28h] [rbp-38h]
  __int64 v89; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v7 = a2;
  *(_QWORD *)(a1 + 352) = a2;
  v8 = *(unsigned int *)(a1 + 400);
  v9 = (__int64)(*(_QWORD *)(a2 + 104) - *(_QWORD *)(a2 + 96)) >> 3;
  if ( (unsigned int)v9 >= v8 )
  {
    if ( (unsigned int)v9 > v8 )
    {
      if ( (unsigned int)v9 > (unsigned __int64)*(unsigned int *)(a1 + 404) )
      {
        sub_16CD150(a1 + 392, (const void *)(a1 + 408), (unsigned int)v9, 16, a5, a6);
        v6 = a1;
        v8 = *(unsigned int *)(a1 + 400);
      }
      v74 = (_QWORD *)(*(_QWORD *)(v6 + 392) + 16 * v8);
      for ( i = *(_QWORD *)(v6 + 392) + 16LL * (unsigned int)v9; (_QWORD *)i != v74; v74 += 2 )
      {
        if ( v74 )
        {
          *v74 = 0;
          v74[1] = 0;
        }
      }
      *(_DWORD *)(v6 + 400) = v9;
      v7 = *(_QWORD *)(v6 + 352);
    }
  }
  else
  {
    *(_DWORD *)(a1 + 400) = v9;
  }
  v10 = *(_QWORD *)(v7 + 328);
  v11 = v7 + 320;
  if ( v7 + 320 != v10 )
  {
    v12 = 0;
    do
    {
      v10 = *(_QWORD *)(v10 + 8);
      ++v12;
    }
    while ( v11 != v10 );
    if ( *(_DWORD *)(v6 + 548) < v12 )
    {
      v87 = v6;
      sub_16CD150(v6 + 536, (const void *)(v6 + 552), v12, 16, v12, a6);
      v6 = v87;
    }
  }
  v88 = v6;
  v13 = sub_1DD57D0(v6, 0, 0);
  v16 = v88;
  v17 = *(_QWORD *)(v88 + 336);
  v89 = v88 + 336;
  *(_QWORD *)(v13 + 8) = v89;
  v17 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v13 = v17 | *(_QWORD *)v13 & 7LL;
  *(_QWORD *)(v17 + 8) = v13;
  v18 = *(_QWORD *)(v16 + 336) & 7LL | v13;
  v19 = *(_QWORD *)(v16 + 352);
  *(_QWORD *)(v16 + 336) = v18;
  v76 = v19 + 320;
  v86 = *(_QWORD *)(v19 + 328);
  if ( v86 == v19 + 320 )
  {
    v36 = *(unsigned int *)(v16 + 544);
    goto LABEL_62;
  }
  v20 = v16;
  v21 = 0;
  v22 = v20;
  while ( 2 )
  {
    v77 = v18 & 0xFFFFFFFFFFFFFFF8LL;
    v23 = *(_QWORD *)(v86 + 32);
    if ( v86 + 24 == v23 )
      goto LABEL_17;
    v24 = v21;
    v25 = v86 + 24;
    do
    {
      while ( 1 )
      {
        if ( (unsigned __int16)(**(_WORD **)(v23 + 16) - 12) <= 1u )
          goto LABEL_14;
        v37 = *(_QWORD *)(v22 + 232);
        v38 = *(_QWORD *)(v22 + 240);
        v24 += 16;
        *(_QWORD *)(v22 + 312) += 32LL;
        if ( ((v37 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v37 + 32 <= v38 - v37 )
        {
          v45 = (v37 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v22 + 232) = v45 + 32;
        }
        else
        {
          v39 = 0x40000000000LL;
          v79 = *(_DWORD *)(v22 + 256);
          if ( v79 >> 7 < 0x1E )
            v39 = 4096LL << (v79 >> 7);
          v40 = malloc(v39);
          v43 = v79;
          if ( !v40 )
          {
            sub_16BD1C0("Allocation failed", 1u);
            v43 = *(unsigned int *)(v22 + 256);
            v40 = 0;
          }
          if ( *(_DWORD *)(v22 + 260) <= (unsigned int)v43 )
          {
            v81 = v40;
            sub_16CD150(v22 + 248, (const void *)(v22 + 264), 0, 8, v41, v42);
            v43 = *(unsigned int *)(v22 + 256);
            v40 = v81;
          }
          v44 = v40 + v39;
          *(_QWORD *)(*(_QWORD *)(v22 + 248) + 8 * v43) = v40;
          v45 = (v40 + 7) & 0xFFFFFFFFFFFFFFF8LL;
          ++*(_DWORD *)(v22 + 256);
          *(_QWORD *)(v22 + 240) = v44;
          *(_QWORD *)(v22 + 232) = v45 + 32;
        }
        *(_QWORD *)(v45 + 16) = v23;
        *(_QWORD *)v45 = 0;
        *(_QWORD *)(v45 + 8) = 0;
        *(_DWORD *)(v45 + 24) = v24;
        v46 = *(_QWORD *)(v22 + 336);
        *(_QWORD *)(v45 + 8) = v89;
        v46 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v45 = v46;
        *(_QWORD *)(v46 + 8) = v45;
        v47 = *(_DWORD *)(v22 + 384);
        v48 = *(_QWORD *)(v22 + 336) & 7LL | v45;
        *(_QWORD *)(v22 + 336) = v48;
        v49 = v48;
        if ( v47 )
        {
          LODWORD(v15) = v47 - 1;
          v14 = *(_QWORD *)(v22 + 368);
          v50 = (v47 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v51 = (_QWORD *)(v14 + 16LL * v50);
          v52 = *v51;
          if ( *v51 == v23 )
            goto LABEL_14;
          v80 = 1;
          v53 = 0;
          while ( v52 != -8 )
          {
            if ( v52 != -16 || v53 )
              v51 = v53;
            v50 = v15 & (v80 + v50);
            v52 = *(_QWORD *)(v14 + 16LL * v50);
            if ( v52 == v23 )
              goto LABEL_14;
            ++v80;
            v53 = v51;
            v51 = (_QWORD *)(v14 + 16LL * v50);
          }
          if ( !v53 )
            v53 = v51;
          v54 = *(_DWORD *)(v22 + 376);
          ++*(_QWORD *)(v22 + 360);
          v55 = v54 + 1;
          if ( 4 * v55 < 3 * v47 )
          {
            if ( v47 - *(_DWORD *)(v22 + 380) - v55 <= v47 >> 3 )
            {
              v84 = v49;
              sub_1DC1390(v22 + 360, v47);
              v69 = *(_DWORD *)(v22 + 384);
              if ( !v69 )
              {
LABEL_98:
                ++*(_DWORD *)(v22 + 376);
                BUG();
              }
              LODWORD(v14) = v69 - 1;
              v70 = 0;
              v49 = v84;
              v71 = 1;
              v72 = v14 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v15 = *(_QWORD *)(v22 + 368);
              v55 = *(_DWORD *)(v22 + 376) + 1;
              v53 = (_QWORD *)(v15 + 16LL * v72);
              v73 = *v53;
              if ( *v53 != v23 )
              {
                while ( v73 != -8 )
                {
                  if ( !v70 && v73 == -16 )
                    v70 = v53;
                  v72 = v14 & (v71 + v72);
                  v53 = (_QWORD *)(v15 + 16LL * v72);
                  v73 = *v53;
                  if ( *v53 == v23 )
                    goto LABEL_48;
                  ++v71;
                }
                if ( v70 )
                  v53 = v70;
              }
            }
            goto LABEL_48;
          }
        }
        else
        {
          ++*(_QWORD *)(v22 + 360);
        }
        v82 = v49;
        sub_1DC1390(v22 + 360, 2 * v47);
        v56 = *(_DWORD *)(v22 + 384);
        if ( !v56 )
          goto LABEL_98;
        LODWORD(v15) = v56 - 1;
        v57 = *(_QWORD *)(v22 + 368);
        v49 = v82;
        v58 = v15 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v55 = *(_DWORD *)(v22 + 376) + 1;
        v53 = (_QWORD *)(v57 + 16LL * v58);
        v14 = *v53;
        if ( *v53 != v23 )
        {
          v59 = 1;
          v60 = 0;
          while ( v14 != -8 )
          {
            if ( v14 == -16 && !v60 )
              v60 = v53;
            v58 = v15 & (v59 + v58);
            v53 = (_QWORD *)(v57 + 16LL * v58);
            v14 = *v53;
            if ( *v53 == v23 )
              goto LABEL_48;
            ++v59;
          }
          if ( v60 )
            v53 = v60;
        }
LABEL_48:
        *(_DWORD *)(v22 + 376) = v55;
        if ( *v53 != -8 )
          --*(_DWORD *)(v22 + 380);
        *v53 = v23;
        v53[1] = v49 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_14:
        if ( (*(_BYTE *)v23 & 4) == 0 )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( v25 == v23 )
          goto LABEL_16;
      }
      while ( (*(_BYTE *)(v23 + 46) & 8) != 0 )
        v23 = *(_QWORD *)(v23 + 8);
      v23 = *(_QWORD *)(v23 + 8);
    }
    while ( v25 != v23 );
LABEL_16:
    v21 = v24;
LABEL_17:
    v26 = *(_QWORD *)(v22 + 232);
    v27 = *(_QWORD *)(v22 + 240);
    v21 += 16;
    *(_QWORD *)(v22 + 312) += 32LL;
    if ( ((v26 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v26 + 32 <= v27 - v26 )
    {
      v32 = (v26 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)(v22 + 232) = v32 + 32;
    }
    else
    {
      v28 = *(unsigned int *)(v22 + 256);
      v78 = v21;
      v29 = 4096LL << ((unsigned int)v28 >> 7);
      if ( (unsigned int)v28 >> 7 >= 0x1E )
        v29 = 0x40000000000LL;
      v30 = malloc(v29);
      v21 = v78;
      v31 = v30;
      if ( !v30 )
      {
        sub_16BD1C0("Allocation failed", 1u);
        v28 = *(unsigned int *)(v22 + 256);
        v21 = v78;
      }
      if ( *(_DWORD *)(v22 + 260) <= (unsigned int)v28 )
      {
        v85 = v21;
        sub_16CD150(v22 + 248, (const void *)(v22 + 264), 0, 8, v14, v15);
        v28 = *(unsigned int *)(v22 + 256);
        v21 = v85;
      }
      *(_QWORD *)(*(_QWORD *)(v22 + 248) + 8 * v28) = v31;
      v32 = (v31 + 7) & 0xFFFFFFFFFFFFFFF8LL;
      ++*(_DWORD *)(v22 + 256);
      *(_QWORD *)(v22 + 240) = v31 + v29;
      *(_QWORD *)(v22 + 232) = v32 + 32;
    }
    *(_QWORD *)v32 = 0;
    *(_QWORD *)(v32 + 8) = 0;
    *(_QWORD *)(v32 + 16) = 0;
    *(_DWORD *)(v32 + 24) = v21;
    v33 = *(_QWORD *)(v22 + 336);
    *(_QWORD *)(v32 + 8) = v89;
    v33 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v32 = v33;
    *(_QWORD *)(v33 + 8) = v32;
    *(_QWORD *)(v22 + 336) = *(_QWORD *)(v22 + 336) & 7LL | v32;
    *(_QWORD *)(*(_QWORD *)(v22 + 392) + 16LL * *(int *)(v86 + 48)) = v77;
    *(_QWORD *)(*(_QWORD *)(v22 + 392) + 16LL * *(int *)(v86 + 48) + 8) = *(_QWORD *)(v22 + 336) & 0xFFFFFFFFFFFFFFF8LL;
    v34 = *(unsigned int *)(v22 + 544);
    if ( (unsigned int)v34 >= *(_DWORD *)(v22 + 548) )
    {
      v83 = v21;
      sub_16CD150(v22 + 536, (const void *)(v22 + 552), 0, 16, v14, v15);
      v34 = *(unsigned int *)(v22 + 544);
      v21 = v83;
    }
    v35 = (unsigned __int64 *)(*(_QWORD *)(v22 + 536) + 16 * v34);
    *v35 = v77;
    v35[1] = v86;
    v36 = (unsigned int)(*(_DWORD *)(v22 + 544) + 1);
    *(_DWORD *)(v22 + 544) = v36;
    v86 = *(_QWORD *)(v86 + 8);
    if ( v76 != v86 )
    {
      v18 = *(_QWORD *)(v22 + 336);
      continue;
    }
    break;
  }
  v16 = v22;
LABEL_62:
  v61 = *(__int64 **)(v16 + 536);
  v62 = 16 * v36;
  v63 = v62;
  v64 = (__int64 *)((char *)v61 + v62);
  if ( (__int64 *)((char *)v61 + v62) != v61 )
  {
    v65 = (unsigned __int64)v61 + v62;
    _BitScanReverse64((unsigned __int64 *)&v62, v62 >> 4);
    sub_1DD9AE0(*(__int64 **)(v16 + 536), v65, 2LL * (int)(63 - (v62 ^ 0x3F)));
    if ( v63 <= 0x100 )
    {
      sub_1F0FDB0(v61, v64);
    }
    else
    {
      v66 = v61 + 32;
      sub_1F0FDB0(v61, v61 + 32);
      if ( v64 != v61 + 32 )
      {
        do
        {
          v67 = v66;
          v66 += 2;
          sub_1F0FD40(v67);
        }
        while ( v64 != v66 );
      }
    }
  }
  return 0;
}
