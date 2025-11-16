// Function: sub_1F0D660
// Address: 0x1f0d660
//
__int64 __fastcall sub_1F0D660(__int64 a1, __int64 a2, __int64 a3)
{
  char *v5; // r12
  char *v6; // r15
  char v7; // al
  __int64 v8; // rbx
  __int16 v9; // ax
  __int64 v10; // rax
  char v11; // al
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  unsigned __int64 v17; // r8
  _DWORD *v18; // r9
  int v19; // r11d
  unsigned __int64 v20; // rax
  __int64 v21; // rdi
  __int64 v22; // rdx
  int v25; // ebx
  char v26; // dl
  __int64 v27; // rdi
  int v28; // esi
  unsigned int v29; // ecx
  unsigned int v30; // ebx
  unsigned int v31; // esi
  int v32; // ebx
  unsigned __int64 v33; // r8
  __int64 v34; // rcx
  unsigned int *v37; // rax
  unsigned int *v38; // rsi
  int v39; // edx
  unsigned int v41; // esi
  unsigned int v42; // eax
  _DWORD *v43; // r10
  int v44; // ecx
  __int64 v45; // rax
  int v46; // eax
  unsigned int v47; // edx
  int v48; // ecx
  __int64 v49; // rdi
  int v50; // eax
  unsigned int v51; // edx
  int v52; // ecx
  int v53; // esi
  int v54; // eax
  int v55; // eax
  int v56; // edi
  _DWORD *v57; // rsi
  const void *v59; // [rsp+18h] [rbp-68h]
  __int64 v60; // [rsp+20h] [rbp-60h]
  unsigned __int64 v61; // [rsp+28h] [rbp-58h]
  int v62; // [rsp+28h] [rbp-58h]
  unsigned __int64 v63[2]; // [rsp+30h] [rbp-50h] BYREF
  int v64; // [rsp+40h] [rbp-40h]

  v5 = *(char **)(a2 + 32);
  v6 = &v5[40 * *(unsigned int *)(a2 + 40)];
  v60 = a1 + 416;
  v59 = (const void *)(a1 + 512);
  if ( v5 == v6 )
    return 0;
  do
  {
    while ( 1 )
    {
      v11 = *v5;
      if ( !*v5 )
      {
        if ( (v5[3] & 0x10) == 0 )
        {
          v7 = v5[4];
          if ( (v7 & 1) != 0 || (v7 & 2) != 0 )
            goto LABEL_16;
        }
        v8 = *((unsigned int *)v5 + 2);
        if ( !(_DWORD)v8 )
          goto LABEL_16;
        v9 = *(_WORD *)(a2 + 46);
        if ( (v9 & 4) != 0 || (v9 & 8) == 0 )
          v10 = (*(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL) >> 4) & 1LL;
        else
          LOBYTE(v10) = sub_1E15D00(a2, 0x10u, 1);
        if ( !(_BYTE)v10 && (_DWORD)v8 == *(_DWORD *)(a1 + 400)
          || (unsigned int)v8 < *(_DWORD *)(a1 + 280) && *(_WORD *)(*(_QWORD *)(a1 + 272) + 2 * v8) )
        {
          return 1;
        }
        goto LABEL_14;
      }
      if ( v11 == 12 )
      {
        v12 = *(unsigned int *)(a1 + 504);
        if ( (_DWORD)v12 )
        {
          v37 = *(unsigned int **)(a1 + 496);
          v38 = &v37[v12];
          goto LABEL_47;
        }
        v13 = *(_QWORD *)(a1 + 576);
        v64 = 0;
        v63[0] = 0;
        v14 = *(_QWORD *)(v13 + 16);
        v63[1] = 0;
        v15 = *(__int64 (**)())(*(_QWORD *)v14 + 48LL);
        if ( v15 == sub_1D90020 )
          BUG();
        v16 = v15();
        (*(void (__fastcall **)(__int64, _QWORD, unsigned __int64 *, __int64))(*(_QWORD *)v16 + 192LL))(
          v16,
          *(_QWORD *)(a1 + 576),
          v63,
          a3);
        v19 = v64;
        if ( v64 )
        {
          v20 = v63[0];
          v21 = (unsigned int)(v64 - 1) >> 6;
          v61 = v63[0];
          v22 = 0;
          while ( 1 )
          {
            _RSI = *(_QWORD *)(v63[0] + 8 * v22);
            if ( v21 == v22 )
              break;
            if ( _RSI )
              goto LABEL_27;
            if ( (_DWORD)v21 + 1 == ++v22 )
              goto LABEL_43;
          }
          _RSI &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v64;
          if ( _RSI )
          {
LABEL_27:
            __asm { tzcnt   rsi, rsi }
            v25 = _RSI + ((_DWORD)v22 << 6);
            if ( v25 != -1 )
            {
              while ( 1 )
              {
                v26 = *(_BYTE *)(a1 + 424) & 1;
                if ( v26 )
                {
                  v27 = a1 + 432;
                  v28 = 15;
                }
                else
                {
                  v41 = *(_DWORD *)(a1 + 440);
                  v27 = *(_QWORD *)(a1 + 432);
                  if ( !v41 )
                  {
                    v42 = *(_DWORD *)(a1 + 424);
                    ++*(_QWORD *)(a1 + 416);
                    v43 = 0;
                    v44 = (v42 >> 1) + 1;
                    goto LABEL_56;
                  }
                  v28 = v41 - 1;
                }
                v29 = v28 & (37 * v25);
                v18 = (_DWORD *)(v27 + 4LL * v29);
                LODWORD(v17) = *v18;
                if ( v25 == *v18 )
                  goto LABEL_31;
                v62 = 1;
                v43 = 0;
                while ( (_DWORD)v17 != -1 )
                {
                  if ( v43 || (_DWORD)v17 != -2 )
                    v18 = v43;
                  v29 = v28 & (v62 + v29);
                  LODWORD(v17) = *(_DWORD *)(v27 + 4LL * v29);
                  if ( v25 == (_DWORD)v17 )
                    goto LABEL_31;
                  ++v62;
                  v43 = v18;
                  v18 = (_DWORD *)(v27 + 4LL * v29);
                }
                v42 = *(_DWORD *)(a1 + 424);
                if ( !v43 )
                  v43 = v18;
                ++*(_QWORD *)(a1 + 416);
                v44 = (v42 >> 1) + 1;
                if ( !v26 )
                {
                  v41 = *(_DWORD *)(a1 + 440);
LABEL_56:
                  if ( 4 * v44 >= 3 * v41 )
                    goto LABEL_70;
                  goto LABEL_57;
                }
                v41 = 16;
                if ( (unsigned int)(4 * v44) >= 0x30 )
                {
LABEL_70:
                  sub_1F0D2C0(v60, 2 * v41);
                  if ( (*(_BYTE *)(a1 + 424) & 1) != 0 )
                  {
                    v17 = a1 + 432;
                    v46 = 15;
                  }
                  else
                  {
                    v54 = *(_DWORD *)(a1 + 440);
                    v17 = *(_QWORD *)(a1 + 432);
                    if ( !v54 )
                      goto LABEL_107;
                    v46 = v54 - 1;
                  }
                  v47 = v46 & (37 * v25);
                  v43 = (_DWORD *)(v17 + 4LL * v47);
                  v48 = *v43;
                  if ( v25 != *v43 )
                  {
                    v56 = 1;
                    v57 = 0;
                    while ( v48 != -1 )
                    {
                      if ( !v57 && v48 == -2 )
                        v57 = v43;
                      LODWORD(v18) = v56 + 1;
                      v47 = v46 & (v56 + v47);
                      v43 = (_DWORD *)(v17 + 4LL * v47);
                      v48 = *v43;
                      if ( v25 == *v43 )
                        goto LABEL_73;
                      ++v56;
                    }
                    if ( v57 )
                    {
                      v42 = *(_DWORD *)(a1 + 424);
                      v43 = v57;
                      goto LABEL_58;
                    }
                  }
                  goto LABEL_73;
                }
LABEL_57:
                if ( v41 - *(_DWORD *)(a1 + 428) - v44 <= v41 >> 3 )
                {
                  sub_1F0D2C0(v60, v41);
                  if ( (*(_BYTE *)(a1 + 424) & 1) != 0 )
                  {
                    v49 = a1 + 432;
                    v50 = 15;
                  }
                  else
                  {
                    v55 = *(_DWORD *)(a1 + 440);
                    v49 = *(_QWORD *)(a1 + 432);
                    if ( !v55 )
                    {
LABEL_107:
                      *(_DWORD *)(a1 + 424) = (2 * (*(_DWORD *)(a1 + 424) >> 1) + 2) | *(_DWORD *)(a1 + 424) & 1;
                      BUG();
                    }
                    v50 = v55 - 1;
                  }
                  v51 = v50 & (37 * v25);
                  v43 = (_DWORD *)(v49 + 4LL * v51);
                  v52 = *v43;
                  if ( v25 != *v43 )
                  {
                    v53 = 1;
                    v17 = 0;
                    while ( v52 != -1 )
                    {
                      if ( !v17 && v52 == -2 )
                        v17 = (unsigned __int64)v43;
                      LODWORD(v18) = v53 + 1;
                      v51 = v50 & (v53 + v51);
                      v43 = (_DWORD *)(v49 + 4LL * v51);
                      v52 = *v43;
                      if ( v25 == *v43 )
                        goto LABEL_73;
                      ++v53;
                    }
                    if ( v17 )
                      v43 = (_DWORD *)v17;
                  }
LABEL_73:
                  v42 = *(_DWORD *)(a1 + 424);
                }
LABEL_58:
                *(_DWORD *)(a1 + 424) = (2 * (v42 >> 1) + 2) | v42 & 1;
                if ( *v43 != -1 )
                  --*(_DWORD *)(a1 + 428);
                *v43 = v25;
                v45 = *(unsigned int *)(a1 + 504);
                if ( (unsigned int)v45 >= *(_DWORD *)(a1 + 508) )
                {
                  sub_16CD150(a1 + 496, v59, 0, 4, v17, (int)v18);
                  v45 = *(unsigned int *)(a1 + 504);
                }
                *(_DWORD *)(*(_QWORD *)(a1 + 496) + 4 * v45) = v25;
                v19 = v64;
                ++*(_DWORD *)(a1 + 504);
                v20 = v63[0];
LABEL_31:
                v30 = v25 + 1;
                v61 = v20;
                if ( v19 != v30 )
                {
                  LODWORD(v18) = v30 >> 6;
                  v31 = (unsigned int)(v19 - 1) >> 6;
                  if ( v30 >> 6 <= v31 )
                  {
                    v32 = v30 & 0x3F;
                    v33 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v32);
                    if ( v32 == 0 )
                      v33 = 0;
                    v34 = (unsigned int)v18;
                    v17 = ~v33;
                    while ( 1 )
                    {
                      _RDX = *(_QWORD *)(v20 + 8 * v34);
                      if ( (_DWORD)v18 == (_DWORD)v34 )
                        _RDX = v17 & *(_QWORD *)(v20 + 8 * v34);
                      if ( v31 == (_DWORD)v34 )
                        _RDX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
                      if ( _RDX )
                        break;
                      if ( v31 < (unsigned int)++v34 )
                        goto LABEL_43;
                    }
                    __asm { tzcnt   rdx, rdx }
                    v25 = _RDX + ((_DWORD)v34 << 6);
                    if ( v25 != -1 )
                      continue;
                  }
                }
                break;
              }
            }
          }
LABEL_43:
          _libc_free(v61);
        }
        else
        {
          _libc_free(v63[0]);
        }
        v37 = *(unsigned int **)(a1 + 496);
        v38 = &v37[*(unsigned int *)(a1 + 504)];
        if ( v37 != v38 )
        {
LABEL_47:
          do
          {
            v39 = *(_DWORD *)(*((_QWORD *)v5 + 3) + 4LL * (*v37 >> 5));
            if ( !_bittest(&v39, *v37) )
              return 1;
            ++v37;
          }
          while ( v38 != v37 );
        }
LABEL_14:
        v11 = *v5;
      }
      if ( v11 == 5 )
        break;
LABEL_16:
      v5 += 40;
      if ( v6 == v5 )
        return 0;
    }
    if ( **(_WORD **)(a2 + 16) != 12 )
      return 1;
    v5 += 40;
  }
  while ( v6 != v5 );
  return 0;
}
