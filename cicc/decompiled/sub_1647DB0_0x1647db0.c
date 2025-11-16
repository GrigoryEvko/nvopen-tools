// Function: sub_1647DB0
// Address: 0x1647db0
//
void __fastcall sub_1647DB0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rbx
  char v4; // al
  unsigned __int8 *v5; // rsi
  int v6; // eax
  unsigned int v7; // esi
  __int64 v8; // rdi
  int v9; // r10d
  __int64 **v10; // r9
  unsigned int v11; // ecx
  __int64 **v12; // rdx
  __int64 *v13; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  __int64 *v17; // r13
  __int64 v18; // rsi
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rdi
  unsigned int v22; // eax
  __int64 *v23; // rsi
  int v24; // r10d
  __int64 **v25; // r8
  int v26; // eax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // r8d
  __int64 **v30; // rdi
  unsigned int v31; // r13d
  __int64 *v32; // rcx

  v3 = (__int64 *)a2;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 != 19 )
  {
LABEL_5:
    if ( (unsigned __int8)(v4 - 4) > 0xCu )
      return;
    v7 = *(_DWORD *)(a1 + 24);
    if ( v7 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      v9 = 1;
      v10 = 0;
      v11 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v12 = (__int64 **)(v8 + 8LL * v11);
      v13 = *v12;
      if ( *v12 == v3 )
        return;
      while ( v13 != (__int64 *)-8LL )
      {
        if ( v13 == (__int64 *)-16LL && !v10 )
          v10 = v12;
        v11 = (v7 - 1) & (v9 + v11);
        v12 = (__int64 **)(v8 + 8LL * v11);
        v13 = *v12;
        if ( *v12 == v3 )
          return;
        ++v9;
      }
      v14 = *(_DWORD *)(a1 + 16);
      if ( !v10 )
        v10 = v12;
      ++*(_QWORD *)a1;
      v15 = v14 + 1;
      if ( 4 * (v14 + 1) < 3 * v7 )
      {
        if ( v7 - *(_DWORD *)(a1 + 20) - v15 > v7 >> 3 )
        {
LABEL_15:
          *(_DWORD *)(a1 + 16) = v15;
          if ( *v10 != (__int64 *)-8LL )
            --*(_DWORD *)(a1 + 20);
          *v10 = v3;
          sub_1647410(a1, *v3);
          if ( *((_BYTE *)v3 + 16) <= 0x17u )
          {
            v16 = 24LL * (*((_DWORD *)v3 + 5) & 0xFFFFFFF);
            v17 = &v3[v16 / 0xFFFFFFFFFFFFFFF8LL];
            if ( (*((_BYTE *)v3 + 23) & 0x40) != 0 )
            {
              v17 = (__int64 *)*(v3 - 1);
              v3 = &v17[(unsigned __int64)v16 / 8];
            }
            while ( v3 != v17 )
            {
              v18 = *v17;
              v17 += 3;
              sub_1647DB0(a1, v18);
            }
          }
          return;
        }
        sub_13B3B90(a1, v7);
        v26 = *(_DWORD *)(a1 + 24);
        if ( v26 )
        {
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = 1;
          v30 = 0;
          v31 = v27 & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
          v10 = (__int64 **)(v28 + 8LL * v31);
          v32 = *v10;
          v15 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v10 != v3 )
          {
            while ( v32 != (__int64 *)-8LL )
            {
              if ( v32 == (__int64 *)-16LL && !v30 )
                v30 = v10;
              v31 = v27 & (v29 + v31);
              v10 = (__int64 **)(v28 + 8LL * v31);
              v32 = *v10;
              if ( *v10 == v3 )
                goto LABEL_15;
              ++v29;
            }
            if ( v30 )
              v10 = v30;
          }
          goto LABEL_15;
        }
LABEL_52:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_13B3B90(a1, 2 * v7);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v19 )
    {
      v20 = v19 - 1;
      v21 = *(_QWORD *)(a1 + 8);
      v22 = (v19 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v10 = (__int64 **)(v21 + 8LL * v22);
      v23 = *v10;
      v15 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v10 != v3 )
      {
        v24 = 1;
        v25 = 0;
        while ( v23 != (__int64 *)-8LL )
        {
          if ( !v25 && v23 == (__int64 *)-16LL )
            v25 = v10;
          v22 = v20 & (v24 + v22);
          v10 = (__int64 **)(v21 + 8LL * v22);
          v23 = *v10;
          if ( *v10 == v3 )
            goto LABEL_15;
          ++v24;
        }
        if ( v25 )
          v10 = v25;
      }
      goto LABEL_15;
    }
    goto LABEL_52;
  }
  while ( 1 )
  {
    v5 = (unsigned __int8 *)v3[3];
    v6 = *v5;
    if ( (unsigned __int8)(v6 - 4) <= 0x1Eu )
      break;
    if ( (unsigned int)(v6 - 1) > 1 )
      return;
    v3 = (__int64 *)*((_QWORD *)v5 + 17);
    v4 = *((_BYTE *)v3 + 16);
    if ( v4 != 19 )
      goto LABEL_5;
  }
  sub_1647B40(a1, (__int64)v5);
}
