// Function: sub_26A95D0
// Address: 0x26a95d0
//
__int64 __fastcall sub_26A95D0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v6; // rax
  unsigned int v7; // esi
  unsigned __int8 *v8; // r13
  __int64 v9; // r9
  __int64 v10; // r8
  int v11; // r11d
  __int64 v12; // rdi
  _QWORD *v13; // r10
  unsigned int v14; // ecx
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rax
  char v18; // dl
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // eax
  __int64 v24; // rsi
  int v25; // edx
  __int64 v26; // rax
  int v27; // eax
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  unsigned int v31; // r15d
  _QWORD *v32; // rdi
  __int64 v33; // rcx

  v6 = sub_BD3990(
         *(unsigned __int8 **)(a3 + 32 * ((*(_BYTE *)(a1 + 241) == 0) + 5LL - (*(_DWORD *)(a3 + 4) & 0x7FFFFFF))),
         a2);
  if ( *v6 )
    return 0;
  v7 = *(_DWORD *)(a1 + 144);
  v8 = v6;
  v9 = a1 + 120;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_13;
  }
  v10 = v7 - 1;
  v11 = 1;
  v12 = *(_QWORD *)(a1 + 128);
  v13 = 0;
  v14 = v10 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (_QWORD *)(v12 + 8LL * v14);
  v16 = *v15;
  if ( a3 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( v13 || v16 != -8192 )
        v15 = v13;
      v14 = v10 & (v11 + v14);
      v16 = *(_QWORD *)(v12 + 8LL * v14);
      if ( a3 == v16 )
        goto LABEL_4;
      ++v11;
      v13 = v15;
      v15 = (_QWORD *)(v12 + 8LL * v14);
    }
    v27 = *(_DWORD *)(a1 + 136);
    if ( !v13 )
      v13 = v15;
    ++*(_QWORD *)(a1 + 120);
    v25 = v27 + 1;
    if ( 4 * (v27 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(a1 + 140) - v25 > v7 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 136) = v25;
        if ( *v13 != -4096 )
          --*(_DWORD *)(a1 + 140);
        *v13 = a3;
        v26 = *(unsigned int *)(a1 + 160);
        if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 164) )
        {
          sub_C8D5F0(a1 + 152, (const void *)(a1 + 168), v26 + 1, 8u, v10, v9);
          v26 = *(unsigned int *)(a1 + 160);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 152) + 8 * v26) = a3;
        ++*(_DWORD *)(a1 + 160);
        goto LABEL_4;
      }
      sub_24FB720(a1 + 120, v7);
      v28 = *(_DWORD *)(a1 + 144);
      if ( v28 )
      {
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 128);
        v10 = 1;
        v31 = v29 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
        v13 = (_QWORD *)(v30 + 8LL * v31);
        v25 = *(_DWORD *)(a1 + 136) + 1;
        v32 = 0;
        v33 = *v13;
        if ( a3 != *v13 )
        {
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v32 )
              v32 = v13;
            v9 = (unsigned int)(v10 + 1);
            v31 = v29 & (v10 + v31);
            v13 = (_QWORD *)(v30 + 8LL * v31);
            v33 = *v13;
            if ( a3 == *v13 )
              goto LABEL_15;
            v10 = (unsigned int)v9;
          }
          if ( v32 )
            v13 = v32;
        }
        goto LABEL_15;
      }
LABEL_51:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_13:
    sub_24FB720(a1 + 120, 2 * v7);
    v20 = *(_DWORD *)(a1 + 144);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 128);
      v23 = (v20 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v13 = (_QWORD *)(v22 + 8LL * v23);
      v24 = *v13;
      v25 = *(_DWORD *)(a1 + 136) + 1;
      if ( a3 != *v13 )
      {
        v9 = 1;
        v10 = 0;
        while ( v24 != -4096 )
        {
          if ( !v10 && v24 == -8192 )
            v10 = (__int64)v13;
          v23 = v21 & (v9 + v23);
          v13 = (_QWORD *)(v22 + 8LL * v23);
          v24 = *v13;
          if ( a3 == *v13 )
            goto LABEL_15;
          v9 = (unsigned int)(v9 + 1);
        }
        if ( v10 )
          v13 = (_QWORD *)v10;
      }
      goto LABEL_15;
    }
    goto LABEL_51;
  }
LABEL_4:
  nullsub_1518();
  v17 = sub_26A73D0(a2, (unsigned __int64)v8 & 0xFFFFFFFFFFFFFFFCLL, 0, a1, 1, 1);
  v18 = 1;
  if ( v17 && !*(_DWORD *)(v17 + 160) )
  {
    v18 = *(_BYTE *)(v17 + 113);
    if ( v18 )
    {
      if ( *(_BYTE *)(v17 + 177) )
        v18 = *(_DWORD *)(v17 + 224) != 0;
    }
    else
    {
      v18 = 1;
    }
  }
  *(_BYTE *)(a1 + 464) |= v18;
  return 1;
}
