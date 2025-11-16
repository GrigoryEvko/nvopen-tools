// Function: sub_1F4E620
// Address: 0x1f4e620
//
void __fastcall sub_1F4E620(__int64 a1, int a2)
{
  char v2; // r14
  int v3; // r12d
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned int v7; // eax
  __int64 v8; // r8
  _BYTE *v9; // r9
  __int64 v10; // rcx
  __int64 i; // rax
  __int64 v12; // r15
  __int16 v13; // ax
  bool v14; // r13
  __int64 *v15; // rax
  __int64 v16; // r8
  char v17; // dl
  __int64 v18; // rax
  __int64 v19; // rsi
  unsigned int v20; // ecx
  __int64 *v21; // rdx
  unsigned int v22; // esi
  int v23; // r13d
  int *v24; // r9
  int *v25; // r8
  int v26; // r15d
  unsigned int v27; // edi
  int *v28; // rdx
  int v29; // eax
  __int64 v30; // rax
  __int64 *v31; // rdi
  __int64 *v32; // rsi
  unsigned int v33; // eax
  unsigned __int64 v34; // rdx
  __int64 v35; // rcx
  unsigned int v36; // eax
  __int64 v37; // r14
  int v38; // ecx
  int v39; // r12d
  char v40; // al
  int v41; // edx
  int v42; // r11d
  int *v43; // rcx
  int v44; // eax
  int v45; // edx
  int v46; // eax
  int v47; // esi
  unsigned int v48; // eax
  int v49; // edi
  int v50; // r10d
  int v51; // eax
  int v52; // eax
  __int64 v53; // rdi
  unsigned int v54; // r15d
  int v55; // esi
  __int64 v56; // rax
  char v58; // [rsp+10h] [rbp-A0h]
  int *v59; // [rsp+10h] [rbp-A0h]
  __int64 v60; // [rsp+18h] [rbp-98h]
  int v61; // [rsp+24h] [rbp-8Ch] BYREF
  int v62; // [rsp+28h] [rbp-88h] BYREF
  int v63; // [rsp+2Ch] [rbp-84h] BYREF
  _BYTE *v64; // [rsp+30h] [rbp-80h] BYREF
  __int64 v65; // [rsp+38h] [rbp-78h]
  _BYTE v66[16]; // [rsp+40h] [rbp-70h] BYREF
  _BYTE v67[96]; // [rsp+50h] [rbp-60h] BYREF

  v2 = 0;
  v3 = a2;
  v64 = v66;
  v65 = 0x400000000LL;
  v61 = 0;
  v60 = a1 + 344;
  while ( 1 )
  {
    v5 = *(_QWORD *)(a1 + 264);
    v6 = *(_QWORD *)(a1 + 304);
    v7 = sub_1E69E00(v5, v3);
    v10 = v7;
    if ( !(_BYTE)v7 )
      goto LABEL_30;
    for ( i = v3 < 0
            ? *(_QWORD *)(*(_QWORD *)(v5 + 24) + 16LL * (v3 & 0x7FFFFFFF) + 8)
            : *(_QWORD *)(*(_QWORD *)(v5 + 272) + 8LL * (unsigned int)v3);
          i && ((*(_BYTE *)(i + 3) & 0x10) != 0 || (*(_BYTE *)(i + 4) & 8) != 0);
          i = *(_QWORD *)(i + 32) )
    {
      ;
    }
    v12 = *(_QWORD *)(i + 16);
    if ( v6 != *(_QWORD *)(v12 + 24) )
    {
LABEL_30:
      v33 = v65;
      goto LABEL_31;
    }
    v61 = 0;
    v13 = **(_WORD **)(v12 + 16);
    if ( v13 == 15 || (v13 & 0xFFFD) == 8 )
    {
      v61 = *(_DWORD *)(*(_QWORD *)(v12 + 32) + 8LL);
      v14 = v61 > 0;
    }
    else
    {
      v58 = v10;
      v40 = sub_1F4C460(v12, v3, &v61, v10, v8, v9);
      LOBYTE(v10) = v58;
      if ( !v40 )
        goto LABEL_30;
      v14 = v61 > 0;
      if ( !v2 )
        goto LABEL_15;
    }
    v15 = *(__int64 **)(a1 + 352);
    if ( *(__int64 **)(a1 + 360) != v15 )
      goto LABEL_14;
    v31 = &v15[*(unsigned int *)(a1 + 372)];
    LODWORD(v16) = *(_DWORD *)(a1 + 372);
    if ( v15 != v31 )
    {
      v32 = 0;
      do
      {
        if ( v12 == *v15 )
          goto LABEL_30;
        if ( *v15 == -2 )
          v32 = v15;
        ++v15;
      }
      while ( v31 != v15 );
      if ( v32 )
      {
        *v32 = v12;
        v2 = v10;
        --*(_DWORD *)(a1 + 376);
        ++*(_QWORD *)(a1 + 344);
        goto LABEL_15;
      }
    }
    if ( (unsigned int)v16 < *(_DWORD *)(a1 + 368) )
    {
      LODWORD(v16) = v16 + 1;
      v2 = v10;
      *(_DWORD *)(a1 + 372) = v16;
      *v31 = v12;
      ++*(_QWORD *)(a1 + 344);
    }
    else
    {
LABEL_14:
      sub_16CCBA0(v60, v12);
      v2 = v17;
      if ( !v17 )
        goto LABEL_30;
    }
LABEL_15:
    v18 = *(unsigned int *)(a1 + 336);
    if ( (_DWORD)v18 )
    {
      v19 = *(_QWORD *)(a1 + 320);
      v20 = (v18 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v21 = (__int64 *)(v19 + 16LL * v20);
      v16 = *v21;
      if ( v12 == *v21 )
      {
LABEL_17:
        if ( v21 != (__int64 *)(v19 + 16 * v18) )
          goto LABEL_30;
      }
      else
      {
        v41 = 1;
        while ( v16 != -8 )
        {
          LODWORD(v9) = v41 + 1;
          v20 = (v18 - 1) & (v41 + v20);
          v21 = (__int64 *)(v19 + 16LL * v20);
          v16 = *v21;
          if ( v12 == *v21 )
            goto LABEL_17;
          v41 = (int)v9;
        }
      }
    }
    if ( v14 )
      break;
    v22 = *(_DWORD *)(a1 + 576);
    v23 = v61;
    if ( v22 )
    {
      LODWORD(v24) = v22 - 1;
      v25 = *(int **)(a1 + 560);
      v26 = 37 * v61;
      v27 = (v22 - 1) & (37 * v61);
      v28 = &v25[2 * v27];
      v29 = *v28;
      if ( v61 == *v28 )
        goto LABEL_21;
      v42 = 1;
      v43 = 0;
      while ( v29 != -1 )
      {
        if ( v29 != -2 || v43 )
          v28 = v43;
        v27 = (unsigned int)v24 & (v42 + v27);
        v59 = &v25[2 * v27];
        v29 = *v59;
        if ( v61 == *v59 )
          goto LABEL_21;
        ++v42;
        v43 = v28;
        v28 = &v25[2 * v27];
      }
      v44 = *(_DWORD *)(a1 + 568);
      if ( !v43 )
        v43 = v28;
      ++*(_QWORD *)(a1 + 552);
      v45 = v44 + 1;
      if ( 4 * (v44 + 1) < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a1 + 572) - v45 <= v22 >> 3 )
        {
          sub_1392B70(a1 + 552, v22);
          v51 = *(_DWORD *)(a1 + 576);
          if ( !v51 )
          {
LABEL_94:
            ++*(_DWORD *)(a1 + 568);
            BUG();
          }
          v52 = v51 - 1;
          v53 = *(_QWORD *)(a1 + 560);
          LODWORD(v24) = 1;
          v25 = 0;
          v54 = v52 & v26;
          v43 = (int *)(v53 + 8LL * v54);
          v55 = *v43;
          v45 = *(_DWORD *)(a1 + 568) + 1;
          if ( v23 != *v43 )
          {
            while ( v55 != -1 )
            {
              if ( v55 == -2 && !v25 )
                v25 = v43;
              v54 = v52 & ((_DWORD)v24 + v54);
              v43 = (int *)(v53 + 8LL * v54);
              v55 = *v43;
              if ( v23 == *v43 )
                goto LABEL_56;
              LODWORD(v24) = (_DWORD)v24 + 1;
            }
            if ( v25 )
              v43 = v25;
          }
        }
        goto LABEL_56;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 552);
    }
    sub_1392B70(a1 + 552, 2 * v22);
    v46 = *(_DWORD *)(a1 + 576);
    if ( !v46 )
      goto LABEL_94;
    v47 = v46 - 1;
    v25 = *(int **)(a1 + 560);
    v48 = (v46 - 1) & (37 * v23);
    v43 = &v25[2 * v48];
    v49 = *v43;
    v45 = *(_DWORD *)(a1 + 568) + 1;
    if ( v23 != *v43 )
    {
      v50 = 1;
      v24 = 0;
      while ( v49 != -1 )
      {
        if ( v49 == -2 && !v24 )
          v24 = v43;
        v48 = v47 & (v50 + v48);
        v43 = &v25[2 * v48];
        v49 = *v43;
        if ( v23 == *v43 )
          goto LABEL_56;
        ++v50;
      }
      if ( v24 )
        v43 = v24;
    }
LABEL_56:
    *(_DWORD *)(a1 + 568) = v45;
    if ( *v43 != -1 )
      --*(_DWORD *)(a1 + 572);
    *v43 = v23;
    v43[1] = v3;
LABEL_21:
    v30 = (unsigned int)v65;
    if ( (unsigned int)v65 >= HIDWORD(v65) )
    {
      sub_16CD150((__int64)&v64, v66, 0, 4, (int)v25, (int)v24);
      v30 = (unsigned int)v65;
    }
    *(_DWORD *)&v64[4 * v30] = v61;
    v3 = v61;
    LODWORD(v65) = v65 + 1;
  }
  v56 = (unsigned int)v65;
  if ( (unsigned int)v65 >= HIDWORD(v65) )
  {
    sub_16CD150((__int64)&v64, v66, 0, 4, v16, (int)v9);
    v56 = (unsigned int)v65;
  }
  *(_DWORD *)&v64[4 * v56] = v61;
  v33 = v65 + 1;
  LODWORD(v65) = v65 + 1;
LABEL_31:
  if ( v33 )
  {
    v34 = (unsigned __int64)v64;
    v35 = v33;
    v36 = v33 - 1;
    v37 = a1 + 584;
    v38 = *(_DWORD *)&v64[4 * v35 - 4];
    LODWORD(v65) = v36;
    if ( v36 )
    {
      while ( 1 )
      {
        v39 = *(_DWORD *)(v34 + 4LL * v36 - 4);
        LODWORD(v65) = v36 - 1;
        v63 = v38;
        v62 = v39;
        sub_1F4E3A0((__int64)v67, v37, &v62, &v63);
        v36 = v65;
        if ( !(_DWORD)v65 )
          break;
        v34 = (unsigned __int64)v64;
        v38 = v39;
      }
    }
    else
    {
      v39 = v38;
    }
    v63 = v39;
    v62 = a2;
    sub_1F4E3A0((__int64)v67, v37, &v62, &v63);
  }
  if ( v64 != v66 )
    _libc_free((unsigned __int64)v64);
}
