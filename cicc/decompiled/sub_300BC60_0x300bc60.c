// Function: sub_300BC60
// Address: 0x300bc60
//
unsigned __int64 __fastcall sub_300BC60(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 i; // rdx
  __int64 v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  __int64 v20; // r13
  __int64 v21; // rbx
  unsigned int v22; // eax
  _DWORD *v23; // rdi
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  _DWORD *v26; // rax
  __int64 v27; // rdx
  _DWORD *v28; // rax
  unsigned int v29; // [rsp+Ch] [rbp-34h]
  unsigned int v30; // [rsp+Ch] [rbp-34h]

  *(_QWORD *)a1 = *(_QWORD *)(a2 + 32);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v4 = 0;
  if ( v3 != sub_2DAC790 )
    v4 = v3();
  *(_QWORD *)(a1 + 8) = v4;
  v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  i = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(a1 + 24) = a2;
  ++*(_QWORD *)(a1 + 104);
  *(_QWORD *)(a1 + 16) = v5;
  *(_DWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  if ( __PAIR64__(*(_DWORD *)(a1 + 124), i) )
  {
    a2 = 64;
    v10 = *(_QWORD *)(a1 + 112);
    v6 = *(unsigned int *)(a1 + 128);
    v11 = 4 * i;
    v12 = 72 * v6;
    if ( (unsigned int)(4 * i) < 0x40 )
      v11 = 64;
    v13 = v10 + v12;
    if ( (unsigned int)v6 <= v11 )
    {
      if ( v13 != v10 )
      {
        do
        {
          v14 = v10 + 72;
          if ( *(_DWORD *)v10 != -1 )
          {
            if ( *(_DWORD *)v10 != -2 )
            {
              v15 = *(_QWORD *)(v10 + 56);
              if ( v15 != v14 )
                _libc_free(v15);
              v16 = *(_QWORD *)(v10 + 40);
              if ( v16 != v10 + 56 )
                _libc_free(v16);
            }
            *(_DWORD *)v10 = -1;
          }
          v10 += 72;
        }
        while ( v14 != v13 );
      }
LABEL_16:
      *(_QWORD *)(a1 + 120) = 0;
      return sub_300BAC0(a1, a2, i, v6, v7, v8);
    }
    do
    {
      v20 = v10 + 72;
      if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
      {
        v18 = *(_QWORD *)(v10 + 56);
        if ( v18 != v20 )
        {
          v29 = i;
          _libc_free(v18);
          i = v29;
        }
        v19 = *(_QWORD *)(v10 + 40);
        if ( v19 != v10 + 56 )
        {
          v30 = i;
          _libc_free(v19);
          i = v30;
        }
      }
      v10 += 72;
    }
    while ( v13 != v20 );
    a2 = *(unsigned int *)(a1 + 128);
    if ( !(_DWORD)i )
    {
      if ( (_DWORD)a2 )
      {
        a2 = v12;
        sub_C7D6A0(*(_QWORD *)(a1 + 112), v12, 8);
        *(_QWORD *)(a1 + 112) = 0;
        *(_QWORD *)(a1 + 120) = 0;
        *(_DWORD *)(a1 + 128) = 0;
        return sub_300BAC0(a1, a2, i, v6, v7, v8);
      }
      goto LABEL_16;
    }
    v21 = 64;
    i = (unsigned int)(i - 1);
    if ( (_DWORD)i )
    {
      _BitScanReverse(&v22, i);
      v6 = 33 - (v22 ^ 0x1F);
      v21 = (unsigned int)(1 << (33 - (v22 ^ 0x1F)));
      if ( (int)v21 < 64 )
        v21 = 64;
    }
    v23 = *(_DWORD **)(a1 + 112);
    if ( (_DWORD)v21 == (_DWORD)a2 )
    {
      *(_QWORD *)(a1 + 120) = 0;
      v28 = &v23[18 * v21];
      do
      {
        if ( v23 )
          *v23 = -1;
        v23 += 18;
      }
      while ( v28 != v23 );
    }
    else
    {
      sub_C7D6A0((__int64)v23, v12, 8);
      a2 = 8;
      v24 = ((((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 16;
      v25 = (v24
           | (((((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v21 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 8)
           | (((((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v21 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v21 / 3u + 1) | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v21 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v21 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 128) = v25;
      v26 = (_DWORD *)sub_C7D670(72 * v25, 8);
      v27 = *(unsigned int *)(a1 + 128);
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 112) = v26;
      for ( i = (__int64)&v26[18 * v27]; (_DWORD *)i != v26; v26 += 18 )
      {
        if ( v26 )
          *v26 = -1;
      }
    }
  }
  return sub_300BAC0(a1, a2, i, v6, v7, v8);
}
