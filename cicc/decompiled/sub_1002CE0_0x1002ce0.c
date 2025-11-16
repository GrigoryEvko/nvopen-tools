// Function: sub_1002CE0
// Address: 0x1002ce0
//
__int64 __fastcall sub_1002CE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // ecx
  __int64 v6; // rdx
  int v7; // r11d
  unsigned int v8; // r8d
  unsigned int i; // eax
  __int64 v10; // r9
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r15
  int v17; // r13d
  unsigned int j; // eax
  __int64 v19; // r10
  unsigned int v20; // eax
  int v21; // r14d
  unsigned int k; // eax
  __int64 v23; // r10
  unsigned int v24; // eax
  int v25; // r11d
  unsigned int m; // eax
  __int64 v27; // r9
  unsigned int v28; // eax
  __int64 v29; // rax

  v5 = *(_DWORD *)(a2 + 88);
  v6 = *(_QWORD *)(a2 + 72);
  if ( v5 )
  {
    v7 = 1;
    v8 = v5 - 1;
    for ( i = (v5 - 1)
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = v8 & v11 )
    {
      v10 = v6 + 24LL * i;
      if ( *(_UNKNOWN **)v10 == &unk_4F81450 && a3 == *(_QWORD *)(v10 + 8) )
        break;
      if ( *(_QWORD *)v10 == -4096 && *(_QWORD *)(v10 + 8) == -4096 )
        goto LABEL_7;
      v11 = v7 + i;
      ++v7;
    }
    v12 = v6 + 24LL * v5;
    if ( v12 != v10 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL);
      if ( v13 )
        v13 += 8;
      goto LABEL_14;
    }
  }
  else
  {
LABEL_7:
    v12 = v6 + 24LL * v5;
    if ( !v5 )
    {
      v13 = 0;
      v14 = 0;
      v15 = 0;
LABEL_9:
      v16 = 0;
      goto LABEL_38;
    }
    v8 = v5 - 1;
  }
  v13 = 0;
LABEL_14:
  v17 = 1;
  for ( j = v8
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F6D3F8 >> 9) ^ ((unsigned int)&unk_4F6D3F8 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v8 & v20 )
  {
    v19 = v6 + 24LL * j;
    if ( *(_UNKNOWN **)v19 == &unk_4F6D3F8 && a3 == *(_QWORD *)(v19 + 8) )
      break;
    if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
      goto LABEL_42;
    v20 = v17 + j;
    ++v17;
  }
  if ( v19 == v12 )
  {
LABEL_42:
    v14 = 0;
    goto LABEL_22;
  }
  v14 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
  if ( v14 )
    v14 += 8;
LABEL_22:
  v21 = 1;
  for ( k = v8
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F86630 >> 9) ^ ((unsigned int)&unk_4F86630 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; k = v8 & v24 )
  {
    v23 = v6 + 24LL * k;
    if ( *(_UNKNOWN **)v23 == &unk_4F86630 && a3 == *(_QWORD *)(v23 + 8) )
      break;
    if ( *(_QWORD *)v23 == -4096 && *(_QWORD *)(v23 + 8) == -4096 )
      goto LABEL_40;
    v24 = v21 + k;
    ++v21;
  }
  if ( v23 == v12 )
  {
LABEL_40:
    v15 = 0;
    goto LABEL_30;
  }
  v15 = *(_QWORD *)(*(_QWORD *)(v23 + 16) + 24LL);
  if ( v15 )
    v15 += 8;
LABEL_30:
  v25 = 1;
  for ( m = v8
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F89C30 >> 9) ^ ((unsigned int)&unk_4F89C30 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; m = v8 & v28 )
  {
    v27 = v6 + 24LL * m;
    if ( *(_UNKNOWN **)v27 == &unk_4F89C30 && a3 == *(_QWORD *)(v27 + 8) )
      break;
    if ( *(_QWORD *)v27 == -4096 && *(_QWORD *)(v27 + 8) == -4096 )
      goto LABEL_9;
    v28 = v25 + m;
    ++v25;
  }
  if ( v12 == v27 )
    goto LABEL_9;
  v16 = *(_QWORD *)(*(_QWORD *)(v27 + 16) + 24LL);
  if ( v16 )
    v16 += 8;
LABEL_38:
  v29 = sub_B2BEC0(a3);
  *(_QWORD *)(a1 + 8) = v14;
  *(_QWORD *)a1 = v29;
  *(_QWORD *)(a1 + 16) = v16;
  *(_QWORD *)(a1 + 24) = v13;
  *(_QWORD *)(a1 + 32) = v15;
  *(_WORD *)(a1 + 64) = 257;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  return a1;
}
