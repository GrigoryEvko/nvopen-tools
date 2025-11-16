// Function: sub_3437D00
// Address: 0x3437d00
//
void __fastcall sub_3437D00(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 *v2; // r14
  unsigned int v4; // esi
  int v5; // r8d
  __int64 *v6; // rdi
  __int64 v7; // r9
  int v8; // r11d
  unsigned int j; // eax
  __int64 *v10; // rdx
  __int64 *v11; // r13
  unsigned int v12; // eax
  int v13; // eax
  int v14; // edx
  int v15; // esi
  int v16; // r11d
  __int64 v17; // r8
  __int64 *v18; // r10
  unsigned int i; // eax
  unsigned int v20; // eax
  int v21; // r15d
  int v22; // eax
  int v23; // edx
  __int64 v24; // rax
  int v25; // eax
  int v26; // edx
  int v27; // esi
  int v28; // r10d
  __int64 v29; // r8
  unsigned int k; // eax
  unsigned int v31; // eax
  int v32; // r9d
  int v33; // r9d

  v1 = *(__int64 **)(a1 + 32);
  v2 = &v1[2 * *(unsigned int *)(a1 + 40)];
  if ( v2 == v1 )
    return;
  do
  {
LABEL_2:
    v4 = *(_DWORD *)(a1 + 24);
    if ( !v4 )
    {
      ++*(_QWORD *)a1;
LABEL_11:
      sub_3437AD0(a1, 2 * v4);
      v13 = *(_DWORD *)(a1 + 24);
      if ( v13 )
      {
        v14 = v13 - 1;
        v15 = *((_DWORD *)v1 + 2);
        v16 = 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = 0;
        for ( i = (v13 - 1) & (v15 + (((unsigned __int64)*v1 >> 9) ^ ((unsigned __int64)*v1 >> 4))); ; i = v14 & v20 )
        {
          v6 = (__int64 *)(v17 + 16LL * i);
          if ( *v1 == *v6 && v15 == *((_DWORD *)v6 + 2) )
            break;
          if ( !*v6 )
          {
            v32 = *((_DWORD *)v6 + 2);
            if ( v32 == -1 )
            {
              if ( v18 )
                v6 = v18;
              v23 = *(_DWORD *)(a1 + 16) + 1;
              goto LABEL_25;
            }
            if ( v32 == -2 && !v18 )
              v18 = (__int64 *)(v17 + 16LL * i);
          }
          v20 = v16 + i;
          ++v16;
        }
        goto LABEL_37;
      }
LABEL_56:
      ++*(_DWORD *)(a1 + 16);
      BUG();
    }
    v5 = *((_DWORD *)v1 + 2);
    v6 = 0;
    v7 = *(_QWORD *)(a1 + 8);
    v8 = 1;
    for ( j = (v4 - 1) & (v5 + (((unsigned __int64)*v1 >> 9) ^ ((unsigned __int64)*v1 >> 4))); ; j = (v4 - 1) & v12 )
    {
      v10 = (__int64 *)(v7 + 16LL * j);
      v11 = (__int64 *)*v10;
      if ( *v1 == *v10 && v5 == *((_DWORD *)v10 + 2) )
      {
        v1 += 2;
        if ( v2 == v1 )
          return;
        goto LABEL_2;
      }
      if ( !v11 )
        break;
LABEL_6:
      v12 = v8 + j;
      ++v8;
    }
    v21 = *((_DWORD *)v10 + 2);
    if ( v21 != -1 )
    {
      if ( v21 == -2 && !v6 )
        v6 = (__int64 *)(v7 + 16LL * j);
      goto LABEL_6;
    }
    v22 = *(_DWORD *)(a1 + 16);
    if ( !v6 )
      v6 = v10;
    ++*(_QWORD *)a1;
    v23 = v22 + 1;
    if ( 4 * (v22 + 1) >= 3 * v4 )
      goto LABEL_11;
    if ( v4 - *(_DWORD *)(a1 + 20) - v23 <= v4 >> 3 )
    {
      sub_3437AD0(a1, v4);
      v25 = *(_DWORD *)(a1 + 24);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = *((_DWORD *)v1 + 2);
        v28 = 1;
        v29 = *(_QWORD *)(a1 + 8);
        for ( k = (v25 - 1) & (v27 + (((unsigned __int64)*v1 >> 9) ^ ((unsigned __int64)*v1 >> 4))); ; k = v26 & v31 )
        {
          v6 = (__int64 *)(v29 + 16LL * k);
          if ( *v1 == *v6 && v27 == *((_DWORD *)v6 + 2) )
            break;
          if ( !*v6 )
          {
            v33 = *((_DWORD *)v6 + 2);
            if ( v33 == -1 )
            {
              if ( v11 )
                v6 = v11;
              v23 = *(_DWORD *)(a1 + 16) + 1;
              goto LABEL_25;
            }
            if ( !v11 && v33 == -2 )
              v11 = (__int64 *)(v29 + 16LL * k);
          }
          v31 = v28 + k;
          ++v28;
        }
LABEL_37:
        v23 = *(_DWORD *)(a1 + 16) + 1;
        goto LABEL_25;
      }
      goto LABEL_56;
    }
LABEL_25:
    *(_DWORD *)(a1 + 16) = v23;
    if ( *v6 || *((_DWORD *)v6 + 2) != -1 )
      --*(_DWORD *)(a1 + 20);
    v24 = *v1;
    v1 += 2;
    *v6 = v24;
    *((_DWORD *)v6 + 2) = *((_DWORD *)v1 - 2);
  }
  while ( v2 != v1 );
}
