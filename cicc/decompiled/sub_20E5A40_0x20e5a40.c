// Function: sub_20E5A40
// Address: 0x20e5a40
//
void __fastcall sub_20E5A40(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r14
  _QWORD *v4; // rax
  int v5; // r12d
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  __int64 v10; // rdx
  __int16 v11; // ax
  __int64 *v12; // r14
  _WORD *v13; // r13
  _WORD *j; // r9
  _QWORD *v15; // r8
  __int64 v16; // r15
  __int64 v17; // r11
  unsigned int v18; // edx
  __int16 v19; // si
  int v20; // eax
  _WORD *v21; // rdx
  _WORD *v22; // rcx
  unsigned __int16 v23; // si
  int v24; // edx
  _WORD *v25; // r10
  unsigned __int16 *v26; // rdx
  int v27; // edi
  __int64 v28; // rax
  unsigned __int16 *v29; // rax
  int v30; // r10d
  unsigned __int16 *v31; // rax
  unsigned __int16 v32; // cx
  unsigned __int16 *k; // rdi
  _QWORD *v34; // r9
  __int64 v35; // r11
  __int64 v36; // r10
  unsigned int v37; // edx
  __int16 v38; // ax
  _WORD *v39; // rdx
  unsigned __int16 v40; // ax
  unsigned __int16 *v41; // rsi
  int v42; // edx
  unsigned __int16 v43; // cx
  int v44; // eax
  unsigned __int16 *v45; // r13
  unsigned __int16 *v46; // rsi
  unsigned __int16 *v47; // rdx
  int v48; // r8d
  __int64 v49; // rax
  unsigned __int16 *v50; // rax
  int v51; // r10d
  __int16 v52; // di
  unsigned __int16 v53; // r8
  __int64 *i; // [rsp+0h] [rbp-60h]
  bool v55; // [rsp+Fh] [rbp-51h]
  unsigned __int64 v56[10]; // [rsp+10h] [rbp-50h] BYREF

  v2 = a2 + 3;
  v4 = (_QWORD *)a2[4];
  if ( v4 == a2 + 3 )
  {
    v5 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v4 = (_QWORD *)v4[1];
      ++v5;
    }
    while ( v4 != v2 );
  }
  v6 = 0;
  v7 = *(unsigned int *)(*(_QWORD *)(a1 + 32) + 16LL);
  if ( (_DWORD)v7 )
  {
    do
    {
      *(_QWORD *)(*(_QWORD *)(a1 + 72) + 2 * v6) = 0;
      *(_DWORD *)(*(_QWORD *)(a1 + 144) + v6) = -1;
      *(_DWORD *)(*(_QWORD *)(a1 + 168) + v6) = v5;
      v6 += 4;
    }
    while ( 4 * v7 != v6 );
  }
  v8 = *(_QWORD *)(a1 + 200);
  if ( v8 )
  {
    memset(*(void **)(a1 + 192), 0, 8 * v8);
    v9 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
    if ( v2 != (_QWORD *)v9 )
      goto LABEL_8;
LABEL_46:
    v55 = 0;
    goto LABEL_12;
  }
  v9 = a2[3] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 == (_QWORD *)v9 )
    goto LABEL_46;
LABEL_8:
  if ( !v9 )
    BUG();
  v10 = *(_QWORD *)v9;
  v11 = *(_WORD *)(v9 + 46);
  if ( (*(_QWORD *)v9 & 4) != 0 )
  {
    if ( (v11 & 4) != 0 )
    {
LABEL_11:
      v55 = (*(_QWORD *)(*(_QWORD *)(v9 + 16) + 8LL) & 8LL) != 0;
      goto LABEL_12;
    }
  }
  else if ( (v11 & 4) != 0 )
  {
    while ( 1 )
    {
      v9 = v10 & 0xFFFFFFFFFFFFFFF8LL;
      v11 = *(_WORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 46);
      if ( (v11 & 4) == 0 )
        break;
      v10 = *(_QWORD *)v9;
    }
  }
  if ( (v11 & 8) == 0 )
    goto LABEL_11;
  v55 = sub_1E15D00(v9, 8u, 1);
LABEL_12:
  v12 = (__int64 *)a2[11];
  for ( i = (__int64 *)a2[12]; i != v12; ++v12 )
  {
    v13 = *(_WORD **)(*v12 + 160);
    for ( j = (_WORD *)sub_1DD77D0(*v12); v13 != j; j += 4 )
    {
      v15 = *(_QWORD **)(a1 + 32);
      if ( !v15 )
        BUG();
      v16 = v15[1];
      v17 = v15[7];
      v18 = *(_DWORD *)(v16 + 24LL * (unsigned __int16)*j + 16);
      v19 = (v18 & 0xF) * *j;
      v20 = 0;
      v21 = (_WORD *)(v17 + 2LL * (v18 >> 4));
      v22 = v21 + 1;
      v23 = *v21 + v19;
      v24 = 0;
      while ( 1 )
      {
        v25 = v22;
        if ( !v22 )
          break;
        while ( 1 )
        {
          v26 = (unsigned __int16 *)(v15[6] + 4LL * v23);
          v27 = *v26;
          v24 = v26[1];
          if ( (_WORD)v27 )
          {
            while ( 1 )
            {
              v28 = v17 + 2LL * *(unsigned int *)(v16 + 24LL * (unsigned __int16)v27 + 8);
              if ( v28 )
                goto LABEL_21;
              if ( !(_WORD)v24 )
                break;
              v27 = v24;
              v24 = 0;
            }
            v20 = v27;
          }
          v52 = *v25;
          v22 = 0;
          ++v25;
          v23 += v52;
          if ( !v52 )
            break;
          v22 = v25;
          if ( !v25 )
            goto LABEL_52;
        }
      }
LABEL_52:
      v27 = v20;
      v28 = 0;
LABEL_21:
      while ( v22 )
      {
        v28 += 2;
        *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * (unsigned __int16)v27) = -1;
        *(_DWORD *)(*(_QWORD *)(a1 + 144) + 4LL * (unsigned __int16)v27) = v5;
        *(_DWORD *)(*(_QWORD *)(a1 + 168) + 4LL * (unsigned __int16)v27) = -1;
        v30 = *(unsigned __int16 *)(v28 - 2);
        v27 += v30;
        if ( !(_WORD)v30 )
        {
          if ( (_WORD)v24 )
          {
            v28 = v15[7] + 2LL * *(unsigned int *)(v15[1] + 24LL * (unsigned __int16)v24 + 8);
            v27 = v24;
            v24 = 0;
          }
          else
          {
            v23 += *v22;
            if ( !*v22 )
              break;
            ++v22;
            v29 = (unsigned __int16 *)(v15[6] + 4LL * v23);
            v27 = *v29;
            v24 = v29[1];
            v28 = v15[7] + 2LL * *(unsigned int *)(v15[1] + 24LL * (unsigned __int16)v27 + 8);
          }
        }
      }
    }
  }
  sub_1E08750((__int64)v56, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL), *(_QWORD *)(a1 + 8));
  v31 = (unsigned __int16 *)sub_1E6A620(*(_QWORD **)(*(_QWORD *)(a1 + 8) + 40LL));
  v32 = *v31;
  for ( k = v31; v32; ++k )
  {
    if ( v55 || (*(_QWORD *)(v56[0] + 8 * ((unsigned __int64)v32 >> 6)) & (1LL << v32)) != 0 )
    {
      v34 = *(_QWORD **)(a1 + 32);
      if ( !v34 )
        BUG();
      v35 = v34[1];
      v36 = v34[7];
      v37 = *(_DWORD *)(v35 + 24LL * v32 + 16);
      v38 = v32 * (v37 & 0xF);
      v39 = (_WORD *)(v36 + 2LL * (v37 >> 4));
      v40 = *v39 + v38;
      v41 = v39 + 1;
      v42 = 0;
      v43 = v40;
      v44 = 0;
LABEL_34:
      v45 = v41;
      while ( 1 )
      {
        v46 = v45;
        if ( !v45 )
        {
          v48 = v44;
          v49 = 0;
          goto LABEL_40;
        }
        v47 = (unsigned __int16 *)(v34[6] + 4LL * v43);
        v48 = *v47;
        v42 = v47[1];
        if ( (_WORD)v48 )
          break;
LABEL_67:
        v53 = *v45;
        v41 = 0;
        ++v45;
        if ( !v53 )
          goto LABEL_34;
        v43 += v53;
      }
      while ( 1 )
      {
        v49 = v36 + 2LL * *(unsigned int *)(v35 + 24LL * (unsigned __int16)v48 + 8);
        if ( v49 )
          break;
        if ( !(_WORD)v42 )
        {
          v44 = v48;
          goto LABEL_67;
        }
        v48 = v42;
        v42 = 0;
      }
LABEL_40:
      while ( v46 )
      {
        v49 += 2;
        *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL * (unsigned __int16)v48) = -1;
        *(_DWORD *)(*(_QWORD *)(a1 + 144) + 4LL * (unsigned __int16)v48) = v5;
        *(_DWORD *)(*(_QWORD *)(a1 + 168) + 4LL * (unsigned __int16)v48) = -1;
        v51 = *(unsigned __int16 *)(v49 - 2);
        v48 += v51;
        if ( !(_WORD)v51 )
        {
          if ( (_WORD)v42 )
          {
            v49 = v34[7] + 2LL * *(unsigned int *)(v34[1] + 24LL * (unsigned __int16)v42 + 8);
            v48 = v42;
            v42 = 0;
          }
          else
          {
            v42 = *v46;
            v43 += v42;
            if ( (_WORD)v42 )
            {
              ++v46;
              v50 = (unsigned __int16 *)(v34[6] + 4LL * v43);
              v48 = *v50;
              v42 = v50[1];
              v49 = v34[7] + 2LL * *(unsigned int *)(v34[1] + 24LL * (unsigned __int16)v48 + 8);
            }
            else
            {
              v49 = 0;
              v46 = 0;
            }
          }
        }
      }
    }
    v32 = k[1];
  }
  _libc_free(v56[0]);
}
