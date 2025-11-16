// Function: sub_1062A30
// Address: 0x1062a30
//
int __fastcall sub_1062A30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  __int64 *v5; // rcx
  __int64 j; // r8
  int v7; // eax
  __int64 v8; // rsi
  __int64 v9; // rdi
  unsigned int v10; // edx
  __int64 *v11; // rax
  __int64 v12; // r10
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  unsigned __int64 v15; // r12
  int v16; // esi
  __int64 v17; // rdx
  __int64 *k; // rdx
  __int64 *v19; // r12
  __int64 *m; // r13
  __int64 v21; // rsi
  __int64 *v22; // rdi
  __int64 *v23; // rdx
  __int64 v24; // rcx
  __int64 *v25; // r12
  __int64 *i; // r13
  __int64 v27; // rdi
  int v28; // eax
  int v29; // r11d

  if ( (unsigned __int8)sub_1062510(a1, a2, a3) )
  {
    v25 = *(__int64 **)(a1 + 40);
    v13 = *(unsigned int *)(a1 + 48);
    for ( i = &v25[v13]; i != v25; ++v25 )
    {
      v27 = *v25;
      if ( *(_BYTE *)(*v25 + 8) == 15 && *(_QWORD *)(v27 + 24) )
        LODWORD(v13) = sub_BCB4B0((__int64 **)v27, byte_3F871B3, 0);
    }
  }
  else
  {
    v5 = *(__int64 **)(a1 + 40);
    for ( j = (__int64)&v5[*(unsigned int *)(a1 + 48)]; (__int64 *)j != v5; ++v5 )
    {
      v7 = *(_DWORD *)(a1 + 32);
      v8 = *v5;
      v9 = *(_QWORD *)(a1 + 16);
      if ( v7 )
      {
        v4 = (unsigned int)(v7 - 1);
        v10 = v4 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v11 = (__int64 *)(v9 + 16LL * v10);
        v12 = *v11;
        if ( v8 == *v11 )
        {
LABEL_5:
          *v11 = -8192;
          --*(_DWORD *)(a1 + 24);
          ++*(_DWORD *)(a1 + 28);
        }
        else
        {
          v28 = 1;
          while ( v12 != -4096 )
          {
            v29 = v28 + 1;
            v10 = v4 & (v28 + v10);
            v11 = (__int64 *)(v9 + 16LL * v10);
            v12 = *v11;
            if ( v8 == *v11 )
              goto LABEL_5;
            v28 = v29;
          }
        }
      }
    }
    v13 = *(unsigned int *)(a1 + 336);
    v14 = *(unsigned int *)(a1 + 192);
    v15 = v13 - v14;
    if ( v13 - v14 != v13 )
    {
      v16 = *(_DWORD *)(a1 + 336) - v14;
      if ( v15 < v13 )
      {
        *(_DWORD *)(a1 + 336) = v16;
      }
      else
      {
        if ( v15 > *(unsigned int *)(a1 + 340) )
        {
          sub_C8D5F0(a1 + 328, (const void *)(a1 + 344), v15, 8u, j, v4);
          v13 = *(unsigned int *)(a1 + 336);
        }
        v17 = *(_QWORD *)(a1 + 328);
        v13 = v17 + 8 * v13;
        for ( k = (__int64 *)(v17 + 8 * v15); k != (__int64 *)v13; v13 += 8LL )
        {
          if ( v13 )
            *(_QWORD *)v13 = 0;
        }
        *(_DWORD *)(a1 + 336) = v16;
        v14 = *(unsigned int *)(a1 + 192);
      }
    }
    v19 = *(__int64 **)(a1 + 184);
    for ( m = &v19[v14]; m != v19; ++v19 )
    {
      v21 = *v19;
      if ( *(_BYTE *)(a1 + 500) )
      {
        v22 = *(__int64 **)(a1 + 480);
        v23 = &v22[*(unsigned int *)(a1 + 492)];
        v13 = (unsigned __int64)v22;
        if ( v22 != v23 )
        {
          while ( v21 != *(_QWORD *)v13 )
          {
            v13 += 8LL;
            if ( v23 == (__int64 *)v13 )
              goto LABEL_23;
          }
          v24 = (unsigned int)(*(_DWORD *)(a1 + 492) - 1);
          *(_DWORD *)(a1 + 492) = v24;
          *(_QWORD *)v13 = v22[v24];
          ++*(_QWORD *)(a1 + 472);
        }
      }
      else
      {
        v13 = (unsigned __int64)sub_C8CA60(a1 + 472, v21);
        if ( v13 )
        {
          *(_QWORD *)v13 = -2;
          ++*(_DWORD *)(a1 + 496);
          ++*(_QWORD *)(a1 + 472);
        }
      }
LABEL_23:
      ;
    }
  }
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 192) = 0;
  return v13;
}
