// Function: sub_35CAC40
// Address: 0x35cac40
//
__int64 __fastcall sub_35CAC40(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  __int64 (*v5)(); // rax
  __int64 v6; // rax
  int v7; // edx
  _BYTE *v8; // rax
  unsigned int v9; // r8d
  _BYTE *v10; // r10
  __int64 v11; // rcx
  unsigned int v14; // ebx
  int v15; // ecx
  __int64 v16; // rdi
  _DWORD *v17; // rcx
  unsigned int *v18; // r8
  __int64 v19; // r9
  _DWORD *v20; // rsi
  unsigned int v21; // ebx
  unsigned int v22; // r11d
  unsigned int v23; // edi
  int v24; // ebx
  __int64 v25; // rsi
  unsigned __int64 v26; // r9
  unsigned __int64 v27; // r9
  unsigned int v30; // esi
  __int64 v31; // r11
  __int64 v32; // r8
  unsigned int *v33; // rdi
  __int64 v34; // r9
  unsigned int *v35; // r10
  int v36; // ecx
  __int64 v37; // rax
  __int64 v38; // rax
  _DWORD *v39; // r13
  _DWORD *v40; // r15
  __int64 v41; // r8
  unsigned int v42; // edx
  _DWORD *v43; // rdi
  int v44; // ecx
  unsigned int v45; // esi
  int v46; // eax
  int v47; // r11d
  __int64 v48; // r9
  unsigned int v49; // edx
  _DWORD *v50; // r10
  int v51; // edi
  int v52; // eax
  int v53; // r11d
  int v54; // eax
  int v55; // eax
  int v56; // r11d
  int v57; // esi
  _DWORD *v58; // rcx
  __int64 v59; // r9
  unsigned int v60; // edx
  int v61; // edi
  int v62; // esi
  int v63; // r8d
  unsigned int v64; // eax
  unsigned int v65; // esi
  int v66; // edi
  unsigned int *v67; // rdx
  int v68; // edx
  int v69; // edx
  int v70; // edi
  unsigned int v71; // r13d
  unsigned int v72; // esi
  unsigned int *v73; // rax
  int v74; // [rsp+10h] [rbp-A0h]
  __int64 v75; // [rsp+18h] [rbp-98h]
  const void *v76; // [rsp+20h] [rbp-90h]
  _QWORD v77[2]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE v78[48]; // [rsp+40h] [rbp-70h] BYREF
  int v79; // [rsp+70h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 648) )
    return a1 + 608;
  v79 = 0;
  v77[0] = v78;
  v77[1] = 0x600000000LL;
  v5 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 720) + 16LL) + 136LL);
  if ( v5 == sub_2DD19D0 )
    BUG();
  v6 = v5();
  v3 = a1 + 608;
  (*(void (__fastcall **)(__int64, _QWORD, _QWORD *, __int64))(*(_QWORD *)v6 + 264LL))(
    v6,
    *(_QWORD *)(a1 + 720),
    v77,
    a2);
  v7 = v79;
  if ( !v79 )
  {
    v10 = (_BYTE *)v77[0];
    goto LABEL_11;
  }
  v8 = (_BYTE *)v77[0];
  v9 = (unsigned int)(v79 - 1) >> 6;
  v10 = (_BYTE *)v77[0];
  v11 = 0;
  while ( 1 )
  {
    _RSI = *(_QWORD *)(v77[0] + 8 * v11);
    if ( v9 == (_DWORD)v11 )
      _RSI = (0xFFFFFFFFFFFFFFFFLL >> -(char)v79) & *(_QWORD *)(v77[0] + 8 * v11);
    if ( _RSI )
      break;
    if ( v9 + 1 == ++v11 )
      goto LABEL_11;
  }
  __asm { tzcnt   rsi, rsi }
  v14 = _RSI + ((_DWORD)v11 << 6);
  if ( v14 != -1 )
  {
    v76 = (const void *)(a1 + 656);
    v75 = a1 + 640;
    while ( 1 )
    {
      v15 = *(_DWORD *)(a1 + 624);
      if ( !v15 )
        break;
      v30 = *(_DWORD *)(a1 + 632);
      if ( !v30 )
      {
        ++*(_QWORD *)(a1 + 608);
        goto LABEL_97;
      }
      v31 = *(_QWORD *)(a1 + 616);
      v32 = (v30 - 1) & (37 * v14);
      v33 = (unsigned int *)(v31 + 4 * v32);
      v34 = *v33;
      if ( v14 != (_DWORD)v34 )
      {
        v74 = 1;
        v35 = 0;
        while ( (_DWORD)v34 != -1 )
        {
          if ( (_DWORD)v34 != -2 || v35 )
            v33 = v35;
          v32 = (v30 - 1) & (v74 + (_DWORD)v32);
          v34 = *(unsigned int *)(v31 + 4LL * (unsigned int)v32);
          if ( v14 == (_DWORD)v34 )
            goto LABEL_24;
          ++v74;
          v35 = v33;
          v33 = (unsigned int *)(v31 + 4LL * (unsigned int)v32);
        }
        if ( !v35 )
          v35 = v33;
        v36 = v15 + 1;
        ++*(_QWORD *)(a1 + 608);
        if ( 4 * v36 < 3 * v30 )
        {
          if ( v30 - *(_DWORD *)(a1 + 628) - v36 <= v30 >> 3 )
          {
            sub_A08C50(a1 + 608, v30);
            v68 = *(_DWORD *)(a1 + 632);
            if ( !v68 )
            {
LABEL_137:
              ++*(_DWORD *)(a1 + 624);
              BUG();
            }
            v69 = v68 - 1;
            v32 = *(_QWORD *)(a1 + 616);
            v70 = 1;
            v71 = v69 & (37 * v14);
            v35 = (unsigned int *)(v32 + 4LL * v71);
            v72 = *v35;
            v36 = *(_DWORD *)(a1 + 624) + 1;
            v73 = 0;
            if ( v14 != *v35 )
            {
              while ( v72 != -1 )
              {
                if ( !v73 && v72 == -2 )
                  v73 = v35;
                v34 = (unsigned int)(v70 + 1);
                v71 = v69 & (v70 + v71);
                v35 = (unsigned int *)(v32 + 4LL * v71);
                v72 = *v35;
                if ( v14 == *v35 )
                  goto LABEL_45;
                ++v70;
              }
              if ( v73 )
                v35 = v73;
            }
          }
LABEL_45:
          *(_DWORD *)(a1 + 624) = v36;
          if ( *v35 != -1 )
            --*(_DWORD *)(a1 + 628);
          *v35 = v14;
          v37 = *(unsigned int *)(a1 + 648);
          if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 652) )
          {
            sub_C8D5F0(v75, v76, v37 + 1, 4u, v32, v34);
            v37 = *(unsigned int *)(a1 + 648);
          }
          *(_DWORD *)(*(_QWORD *)(a1 + 640) + 4 * v37) = v14;
          ++*(_DWORD *)(a1 + 648);
LABEL_57:
          v7 = v79;
          v8 = (_BYTE *)v77[0];
          goto LABEL_24;
        }
LABEL_97:
        sub_A08C50(a1 + 608, 2 * v30);
        v63 = *(_DWORD *)(a1 + 632);
        if ( !v63 )
          goto LABEL_137;
        v32 = (unsigned int)(v63 - 1);
        v34 = *(_QWORD *)(a1 + 616);
        v64 = v32 & (37 * v14);
        v35 = (unsigned int *)(v34 + 4LL * v64);
        v65 = *v35;
        v36 = *(_DWORD *)(a1 + 624) + 1;
        if ( v14 != *v35 )
        {
          v66 = 1;
          v67 = 0;
          while ( v65 != -1 )
          {
            if ( v65 == -2 && !v67 )
              v67 = v35;
            v64 = v32 & (v66 + v64);
            v35 = (unsigned int *)(v34 + 4LL * v64);
            v65 = *v35;
            if ( v14 == *v35 )
              goto LABEL_45;
            ++v66;
          }
          if ( v67 )
            v35 = v67;
        }
        goto LABEL_45;
      }
LABEL_24:
      v21 = v14 + 1;
      v10 = v8;
      if ( v7 != v21 )
      {
        v22 = v21 >> 6;
        v23 = (unsigned int)(v7 - 1) >> 6;
        if ( v21 >> 6 <= v23 )
        {
          v24 = v21 & 0x3F;
          v25 = v22;
          v26 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v24);
          if ( v24 == 0 )
            v26 = 0;
          v27 = ~v26;
          while ( 1 )
          {
            _RCX = *(_QWORD *)&v8[8 * v25];
            if ( v22 == (_DWORD)v25 )
              _RCX = v27 & *(_QWORD *)&v8[8 * v25];
            if ( (_DWORD)v25 == v23 )
              _RCX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v7;
            if ( _RCX )
              break;
            if ( v23 < (unsigned int)++v25 )
              goto LABEL_36;
          }
          __asm { tzcnt   rcx, rcx }
          v14 = _RCX + ((_DWORD)v25 << 6);
          if ( v14 != -1 )
            continue;
        }
      }
LABEL_36:
      v3 = a1 + 608;
      goto LABEL_11;
    }
    v16 = *(unsigned int *)(a1 + 648);
    v17 = *(_DWORD **)(a1 + 640);
    v18 = &v17[v16];
    v19 = (4 * v16) >> 2;
    if ( !((4 * v16) >> 4) )
      goto LABEL_51;
    v20 = &v17[4 * ((4 * v16) >> 4)];
    do
    {
      if ( v14 == *v17 )
        goto LABEL_23;
      if ( v14 == v17[1] )
      {
        ++v17;
        goto LABEL_23;
      }
      if ( v14 == v17[2] )
      {
        v17 += 2;
        goto LABEL_23;
      }
      if ( v14 == v17[3] )
      {
        v17 += 3;
        goto LABEL_23;
      }
      v17 += 4;
    }
    while ( v17 != v20 );
    v19 = v18 - v17;
LABEL_51:
    if ( v19 == 2 )
      goto LABEL_60;
    if ( v19 == 3 )
    {
      if ( v14 != *v17 )
      {
        ++v17;
LABEL_60:
        if ( v14 != *v17 )
        {
          ++v17;
          goto LABEL_62;
        }
      }
LABEL_23:
      if ( v18 == v17 )
        goto LABEL_54;
      goto LABEL_24;
    }
    if ( v19 != 1 )
      goto LABEL_54;
LABEL_62:
    if ( v14 == *v17 )
      goto LABEL_23;
LABEL_54:
    if ( v16 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 652) )
    {
      sub_C8D5F0(v75, v76, v16 + 1, 4u, (__int64)v18, v19);
      v18 = (unsigned int *)(*(_QWORD *)(a1 + 640) + 4LL * *(unsigned int *)(a1 + 648));
    }
    *v18 = v14;
    v38 = (unsigned int)(*(_DWORD *)(a1 + 648) + 1);
    *(_DWORD *)(a1 + 648) = v38;
    if ( (unsigned int)v38 <= 0x10 )
      goto LABEL_57;
    v39 = *(_DWORD **)(a1 + 640);
    v40 = &v39[v38];
    while ( 1 )
    {
      v45 = *(_DWORD *)(a1 + 632);
      if ( !v45 )
        break;
      v41 = *(_QWORD *)(a1 + 616);
      v42 = (v45 - 1) & (37 * *v39);
      v43 = (_DWORD *)(v41 + 4LL * v42);
      v44 = *v43;
      if ( *v39 != *v43 )
      {
        v53 = 1;
        v50 = 0;
        while ( v44 != -1 )
        {
          if ( v44 != -2 || v50 )
            v43 = v50;
          v42 = (v45 - 1) & (v53 + v42);
          v44 = *(_DWORD *)(v41 + 4LL * v42);
          if ( *v39 == v44 )
            goto LABEL_69;
          ++v53;
          v50 = v43;
          v43 = (_DWORD *)(v41 + 4LL * v42);
        }
        v54 = *(_DWORD *)(a1 + 624);
        if ( !v50 )
          v50 = v43;
        ++*(_QWORD *)(a1 + 608);
        v52 = v54 + 1;
        if ( 4 * v52 < 3 * v45 )
        {
          if ( v45 - *(_DWORD *)(a1 + 628) - v52 <= v45 >> 3 )
          {
            sub_A08C50(a1 + 608, v45);
            v55 = *(_DWORD *)(a1 + 632);
            if ( !v55 )
            {
LABEL_138:
              ++*(_DWORD *)(a1 + 624);
              BUG();
            }
            v56 = v55 - 1;
            v57 = 1;
            v58 = 0;
            v59 = *(_QWORD *)(a1 + 616);
            v60 = (v55 - 1) & (37 * *v39);
            v50 = (_DWORD *)(v59 + 4LL * v60);
            v61 = *v50;
            v52 = *(_DWORD *)(a1 + 624) + 1;
            if ( *v39 != *v50 )
            {
              while ( v61 != -1 )
              {
                if ( v61 == -2 && !v58 )
                  v58 = v50;
                v60 = v56 & (v57 + v60);
                v50 = (_DWORD *)(v59 + 4LL * v60);
                v61 = *v50;
                if ( *v39 == *v50 )
                  goto LABEL_74;
                ++v57;
              }
LABEL_86:
              if ( v58 )
                v50 = v58;
            }
          }
LABEL_74:
          *(_DWORD *)(a1 + 624) = v52;
          if ( *v50 != -1 )
            --*(_DWORD *)(a1 + 628);
          *v50 = *v39;
          goto LABEL_69;
        }
LABEL_72:
        sub_A08C50(a1 + 608, 2 * v45);
        v46 = *(_DWORD *)(a1 + 632);
        if ( !v46 )
          goto LABEL_138;
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 616);
        v49 = (v46 - 1) & (37 * *v39);
        v50 = (_DWORD *)(v48 + 4LL * v49);
        v51 = *v50;
        v52 = *(_DWORD *)(a1 + 624) + 1;
        if ( *v39 != *v50 )
        {
          v62 = 1;
          v58 = 0;
          while ( v51 != -1 )
          {
            if ( !v58 && v51 == -2 )
              v58 = v50;
            v49 = v47 & (v62 + v49);
            v50 = (_DWORD *)(v48 + 4LL * v49);
            v51 = *v50;
            if ( *v39 == *v50 )
              goto LABEL_74;
            ++v62;
          }
          goto LABEL_86;
        }
        goto LABEL_74;
      }
LABEL_69:
      if ( v40 == ++v39 )
        goto LABEL_57;
    }
    ++*(_QWORD *)(a1 + 608);
    goto LABEL_72;
  }
LABEL_11:
  if ( v10 != v78 )
    _libc_free((unsigned __int64)v10);
  return v3;
}
