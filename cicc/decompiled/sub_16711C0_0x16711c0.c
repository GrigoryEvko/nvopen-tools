// Function: sub_16711C0
// Address: 0x16711c0
//
void __fastcall sub_16711C0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rdx
  _QWORD *j; // r9
  int v6; // eax
  int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // r10
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // r12d
  int v15; // r14d
  unsigned __int64 v16; // r13
  _QWORD *v17; // r15
  _QWORD *v18; // r12
  _QWORD *v19; // rax
  __int64 v20; // rax
  __int64 v21; // r14
  _QWORD *v22; // rdx
  __int64 *v23; // r12
  __int64 *i; // r13
  __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *v27; // rax
  _QWORD *k; // rdx
  __int64 v29; // rdx
  __int64 v30; // rcx
  int v31; // eax
  int v32; // r11d

  if ( (unsigned __int8)sub_1670BE0(a1, a2, a3) )
  {
    v23 = *(__int64 **)(a1 + 40);
    for ( i = &v23[*(unsigned int *)(a1 + 48)]; i != v23; ++v23 )
    {
      v25 = *v23;
      if ( *(_BYTE *)(*v23 + 8) == 13 && *(_QWORD *)(v25 + 24) )
        sub_1643660((__int64 **)v25, byte_3F871B3, 0);
    }
    goto LABEL_21;
  }
  v4 = *(_QWORD **)(a1 + 40);
  for ( j = &v4[*(unsigned int *)(a1 + 48)]; v4 != j; ++v4 )
  {
    v6 = *(_DWORD *)(a1 + 32);
    if ( v6 )
    {
      v7 = v6 - 1;
      v8 = *(_QWORD *)(a1 + 16);
      v9 = (v6 - 1) & (((unsigned int)*v4 >> 9) ^ ((unsigned int)*v4 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( *v4 == *v10 )
      {
LABEL_5:
        *v10 = -16;
        --*(_DWORD *)(a1 + 24);
        ++*(_DWORD *)(a1 + 28);
      }
      else
      {
        v31 = 1;
        while ( v11 != -8 )
        {
          v32 = v31 + 1;
          v9 = v7 & (v31 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v4 == *v10 )
            goto LABEL_5;
          v31 = v32;
        }
      }
    }
  }
  v12 = *(unsigned int *)(a1 + 336);
  v13 = *(unsigned int *)(a1 + 192);
  v14 = *(_DWORD *)(a1 + 336);
  v15 = *(_DWORD *)(a1 + 192);
  v16 = v12 - v13;
  if ( v12 < v13 )
  {
    if ( v16 > *(unsigned int *)(a1 + 340) )
    {
      sub_16CD150(a1 + 328, a1 + 344, v16, 8);
      v12 = *(unsigned int *)(a1 + 336);
    }
    v26 = *(_QWORD *)(a1 + 328);
    v27 = (_QWORD *)(v26 + 8 * v12);
    for ( k = (_QWORD *)(v26 + 8 * v16); k != v27; ++v27 )
    {
      if ( v27 )
        *v27 = 0;
    }
    v13 = *(unsigned int *)(a1 + 192);
    *(_DWORD *)(a1 + 336) = v14 - v15;
  }
  else
  {
    *(_DWORD *)(a1 + 336) = v14 - v13;
  }
  v17 = *(_QWORD **)(a1 + 184);
  v18 = &v17[v13];
  while ( v18 != v17 )
  {
    while ( 1 )
    {
      v21 = *v17;
      v19 = *(_QWORD **)(a1 + 480);
      if ( *(_QWORD **)(a1 + 488) != v19 )
        break;
      v22 = &v19[*(unsigned int *)(a1 + 500)];
      if ( v19 == v22 )
      {
LABEL_38:
        v19 = v22;
      }
      else
      {
        while ( v21 != *v19 )
        {
          if ( v22 == ++v19 )
            goto LABEL_38;
        }
      }
LABEL_19:
      if ( v22 == v19 )
        goto LABEL_13;
      ++v17;
      *v19 = -2;
      ++*(_DWORD *)(a1 + 504);
      if ( v18 == v17 )
        goto LABEL_21;
    }
    v19 = (_QWORD *)sub_16CC9F0(a1 + 472, *v17);
    if ( v21 == *v19 )
    {
      v29 = *(_QWORD *)(a1 + 488);
      if ( v29 == *(_QWORD *)(a1 + 480) )
        v30 = *(unsigned int *)(a1 + 500);
      else
        v30 = *(unsigned int *)(a1 + 496);
      v22 = (_QWORD *)(v29 + 8 * v30);
      goto LABEL_19;
    }
    v20 = *(_QWORD *)(a1 + 488);
    if ( v20 == *(_QWORD *)(a1 + 480) )
    {
      v19 = (_QWORD *)(v20 + 8LL * *(unsigned int *)(a1 + 500));
      v22 = v19;
      goto LABEL_19;
    }
LABEL_13:
    ++v17;
  }
LABEL_21:
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 192) = 0;
}
