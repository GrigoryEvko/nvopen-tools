// Function: sub_31BE9C0
// Address: 0x31be9c0
//
__int64 __fastcall sub_31BE9C0(__int64 a1)
{
  int v2; // edx
  unsigned int v3; // edx
  __int64 v4; // rax
  _QWORD *v5; // rbx
  __int64 v6; // rax
  _QWORD *i; // r14
  unsigned __int64 v8; // r13
  __int64 *v9; // rax
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  int v12; // edx
  unsigned int v13; // edx
  __int64 v14; // rax
  _QWORD *v15; // rbx
  __int64 v16; // rax
  _QWORD *j; // r13
  __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // r12
  int v21; // eax
  __int64 v22; // rdx
  _QWORD *v23; // rax
  _QWORD *k; // rdx
  __int64 result; // rax
  unsigned int v26; // eax
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 n; // r13
  unsigned int v30; // ecx
  unsigned int v31; // eax
  _QWORD *v32; // rdi
  int v33; // ebx
  _QWORD *v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rdi
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *m; // rdx

  v2 = *(_DWORD *)(a1 + 232);
  ++*(_QWORD *)(a1 + 216);
  if ( v2 || *(_DWORD *)(a1 + 236) )
  {
    v3 = 4 * v2;
    v4 = *(unsigned int *)(a1 + 240);
    if ( v3 < 0x40 )
      v3 = 64;
    if ( (unsigned int)v4 > v3 )
    {
      sub_31BE760(a1 + 216);
    }
    else
    {
      v5 = *(_QWORD **)(a1 + 224);
      v6 = 2 * v4;
      for ( i = &v5[v6]; i != v5; v5 += 2 )
      {
        if ( *v5 != -4096 )
        {
          if ( *v5 != -8192 )
          {
            v8 = v5[1];
            if ( v8 )
            {
              v9 = *(__int64 **)v8;
              v10 = *(_QWORD *)v8 + 8LL * *(unsigned int *)(v8 + 8);
              if ( *(_QWORD *)v8 != v10 )
              {
                do
                {
                  v11 = *v9++;
                  *(_QWORD *)(v11 + 32) = 0;
                }
                while ( (__int64 *)v10 != v9 );
                v10 = *(_QWORD *)v8;
              }
              if ( v10 != v8 + 16 )
                _libc_free(v10);
              j_j___libc_free_0(v8);
            }
          }
          *v5 = -4096;
        }
      }
      *(_QWORD *)(a1 + 232) = 0;
    }
  }
  v12 = *(_DWORD *)(a1 + 56);
  ++*(_QWORD *)(a1 + 40);
  if ( v12 || *(_DWORD *)(a1 + 60) )
  {
    v13 = 4 * v12;
    v14 = *(unsigned int *)(a1 + 64);
    if ( v13 < 0x40 )
      v13 = 64;
    if ( (unsigned int)v14 > v13 )
    {
      sub_31BE530(a1 + 40);
    }
    else
    {
      v15 = *(_QWORD **)(a1 + 48);
      v16 = 2 * v14;
      for ( j = &v15[v16]; j != v15; v15 += 2 )
      {
        if ( *v15 != -4096 )
        {
          if ( *v15 != -8192 )
          {
            v18 = v15[1];
            if ( v18 )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)v18 + 8LL))(v18);
          }
          *v15 = -4096;
        }
      }
      *(_QWORD *)(a1 + 56) = 0;
    }
  }
  v19 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v19 )
    j_j___libc_free_0(v19);
  if ( *(_BYTE *)(a1 + 208) )
    *(_BYTE *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  v20 = *(_QWORD *)(a1 + 352);
  v21 = *(_DWORD *)(v20 + 16);
  ++*(_QWORD *)v20;
  if ( !v21 )
  {
    if ( !*(_DWORD *)(v20 + 20) )
      goto LABEL_42;
    v22 = *(unsigned int *)(v20 + 24);
    if ( (unsigned int)v22 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(v20 + 8), 16LL * (unsigned int)v22, 8);
      *(_QWORD *)(v20 + 8) = 0;
      *(_QWORD *)(v20 + 16) = 0;
      *(_DWORD *)(v20 + 24) = 0;
      goto LABEL_42;
    }
    goto LABEL_39;
  }
  v30 = 4 * v21;
  v22 = *(unsigned int *)(v20 + 24);
  if ( (unsigned int)(4 * v21) < 0x40 )
    v30 = 64;
  if ( (unsigned int)v22 <= v30 )
  {
LABEL_39:
    v23 = *(_QWORD **)(v20 + 8);
    for ( k = &v23[2 * v22]; k != v23; v23 += 2 )
      *v23 = -4096;
    *(_QWORD *)(v20 + 16) = 0;
    goto LABEL_42;
  }
  v31 = v21 - 1;
  if ( v31 )
  {
    _BitScanReverse(&v31, v31);
    v32 = *(_QWORD **)(v20 + 8);
    v33 = 1 << (33 - (v31 ^ 0x1F));
    if ( v33 < 64 )
      v33 = 64;
    if ( (_DWORD)v22 == v33 )
    {
      *(_QWORD *)(v20 + 16) = 0;
      v34 = &v32[2 * (unsigned int)v22];
      do
      {
        if ( v32 )
          *v32 = -4096;
        v32 += 2;
      }
      while ( v34 != v32 );
      goto LABEL_42;
    }
  }
  else
  {
    v32 = *(_QWORD **)(v20 + 8);
    v33 = 64;
  }
  sub_C7D6A0((__int64)v32, 16LL * (unsigned int)v22, 8);
  v35 = ((((((((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
           | (4 * v33 / 3u + 1)
           | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 4)
         | (((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
         | (4 * v33 / 3u + 1)
         | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
         | (4 * v33 / 3u + 1)
         | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 4)
       | (((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
       | (4 * v33 / 3u + 1)
       | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 16;
  v36 = (v35
       | (((((((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
           | (4 * v33 / 3u + 1)
           | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 4)
         | (((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
         | (4 * v33 / 3u + 1)
         | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
         | (4 * v33 / 3u + 1)
         | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 4)
       | (((4 * v33 / 3u + 1) | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1)) >> 2)
       | (4 * v33 / 3u + 1)
       | ((unsigned __int64)(4 * v33 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(v20 + 24) = v36;
  v37 = (_QWORD *)sub_C7D670(16 * v36, 8);
  v38 = *(unsigned int *)(v20 + 24);
  *(_QWORD *)(v20 + 16) = 0;
  *(_QWORD *)(v20 + 8) = v37;
  for ( m = &v37[2 * v38]; m != v37; v37 += 2 )
  {
    if ( v37 )
      *v37 = -4096;
  }
LABEL_42:
  result = *(unsigned int *)(v20 + 48);
  ++*(_QWORD *)(v20 + 32);
  if ( __PAIR64__(*(_DWORD *)(v20 + 52), result) )
  {
    v26 = 4 * result;
    v27 = *(unsigned int *)(v20 + 56);
    if ( v26 < 0x40 )
      v26 = 64;
    if ( (unsigned int)v27 > v26 )
    {
      return sub_31BE2F0(v20 + 32);
    }
    else
    {
      v28 = *(_QWORD *)(v20 + 40);
      result = 5 * v27;
      for ( n = v28 + 40 * v27; n != v28; v28 += 40 )
      {
        result = *(_QWORD *)v28;
        if ( *(_QWORD *)v28 != -4096 )
        {
          if ( result != -8192 )
            result = sub_C7D6A0(*(_QWORD *)(v28 + 16), 16LL * *(unsigned int *)(v28 + 32), 8);
          *(_QWORD *)v28 = -4096;
        }
      }
      *(_QWORD *)(v20 + 48) = 0;
    }
  }
  return result;
}
