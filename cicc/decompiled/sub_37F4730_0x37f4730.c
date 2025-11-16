// Function: sub_37F4730
// Address: 0x37f4730
//
__int64 __fastcall sub_37F4730(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  unsigned __int64 *v6; // rcx
  unsigned __int64 *v7; // r12
  unsigned __int64 v8; // rdi
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // rax
  unsigned __int64 *v12; // rax
  unsigned __int64 v13; // r15
  int v14; // r15d
  __int64 v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // r14
  __int64 v18; // r12
  unsigned __int64 v19; // rdi
  int v20; // eax
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *j; // rdx
  __int64 result; // rax
  unsigned int v25; // ecx
  unsigned int v26; // eax
  _QWORD *v27; // rdi
  int v28; // ebx
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // rdx
  int v32; // ebx
  unsigned int v33; // eax
  _DWORD *v34; // rdi
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _DWORD *v37; // rax
  __int64 v38; // rdx
  _DWORD *i; // rdx
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 v43; // rdx
  _QWORD *k; // rdx
  _DWORD *v45; // rax
  unsigned __int64 *v46; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 344);
  v3 = v2 + 24LL * *(unsigned int *)(a1 + 352);
  while ( v2 != v3 )
  {
    while ( 1 )
    {
      v4 = *(_QWORD *)(v3 - 24);
      v3 -= 24;
      if ( !v4 )
        break;
      j_j___libc_free_0(v4);
      if ( v2 == v3 )
        goto LABEL_5;
    }
  }
LABEL_5:
  v5 = *(unsigned int *)(a1 + 504);
  v6 = *(unsigned __int64 **)(a1 + 496);
  *(_DWORD *)(a1 + 352) = 0;
  v46 = v6;
  v7 = &v6[3 * v5];
  if ( v6 != v7 )
  {
    do
    {
      v8 = *(v7 - 3);
      v9 = (__int64 *)*(v7 - 2);
      v7 -= 3;
      v10 = (__int64 *)v8;
      if ( v9 != (__int64 *)v8 )
      {
        do
        {
          v11 = *v10;
          if ( *v10 )
          {
            if ( (v11 & 1) != 0 )
            {
              v12 = (unsigned __int64 *)(v11 & 0xFFFFFFFFFFFFFFFELL);
              v13 = (unsigned __int64)v12;
              if ( v12 )
              {
                if ( (unsigned __int64 *)*v12 != v12 + 2 )
                  _libc_free(*v12);
                j_j___libc_free_0(v13);
              }
            }
          }
          ++v10;
        }
        while ( v9 != v10 );
        v8 = *v7;
      }
      if ( v8 )
        j_j___libc_free_0(v8);
    }
    while ( v46 != v7 );
  }
  v14 = *(_DWORD *)(a1 + 624);
  ++*(_QWORD *)(a1 + 608);
  *(_DWORD *)(a1 + 504) = 0;
  if ( v14 || *(_DWORD *)(a1 + 628) )
  {
    v15 = *(_QWORD *)(a1 + 616);
    v16 = 4 * v14;
    v17 = 72LL * *(unsigned int *)(a1 + 632);
    if ( (unsigned int)(4 * v14) < 0x40 )
      v16 = 64;
    v18 = v15 + v17;
    if ( *(_DWORD *)(a1 + 632) <= v16 )
    {
      if ( v18 == v15 )
        goto LABEL_32;
      while ( 1 )
      {
        while ( *(_DWORD *)v15 == -1 )
        {
          if ( *(_DWORD *)(v15 + 4) != 0x7FFFFFFF )
            goto LABEL_26;
          v15 += 72;
          if ( v15 == v18 )
            goto LABEL_32;
        }
        if ( *(_DWORD *)v15 != -2 || *(_DWORD *)(v15 + 4) != 0x80000000 )
        {
LABEL_26:
          v19 = *(_QWORD *)(v15 + 8);
          if ( v19 != v15 + 24 )
            _libc_free(v19);
        }
        *(_DWORD *)v15 = -1;
        v15 += 72;
        *(_DWORD *)(v15 - 68) = 0x7FFFFFFF;
        if ( v15 == v18 )
          goto LABEL_32;
      }
    }
    while ( 1 )
    {
      if ( *(_DWORD *)v15 == -1 )
      {
        if ( *(_DWORD *)(v15 + 4) != 0x7FFFFFFF )
          goto LABEL_57;
      }
      else if ( *(_DWORD *)v15 != -2 || *(_DWORD *)(v15 + 4) != 0x80000000 )
      {
LABEL_57:
        v30 = *(_QWORD *)(v15 + 8);
        if ( v30 != v15 + 24 )
          _libc_free(v30);
      }
      v15 += 72;
      if ( v15 == v18 )
      {
        v31 = *(unsigned int *)(a1 + 632);
        if ( v14 )
        {
          v32 = 64;
          if ( v14 != 1 )
          {
            _BitScanReverse(&v33, v14 - 1);
            v32 = 1 << (33 - (v33 ^ 0x1F));
            if ( v32 < 64 )
              v32 = 64;
          }
          v34 = *(_DWORD **)(a1 + 616);
          if ( (_DWORD)v31 == v32 )
          {
            *(_QWORD *)(a1 + 624) = 0;
            v45 = &v34[18 * v31];
            do
            {
              if ( v34 )
              {
                *v34 = -1;
                v34[1] = 0x7FFFFFFF;
              }
              v34 += 18;
            }
            while ( v45 != v34 );
          }
          else
          {
            sub_C7D6A0((__int64)v34, v17, 8);
            v35 = ((((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                     | (4 * v32 / 3u + 1)
                     | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                   | (4 * v32 / 3u + 1)
                   | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                   | (4 * v32 / 3u + 1)
                   | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                 | (4 * v32 / 3u + 1)
                 | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 16;
            v36 = (v35
                 | (((((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                     | (4 * v32 / 3u + 1)
                     | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                   | (4 * v32 / 3u + 1)
                   | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                   | (4 * v32 / 3u + 1)
                   | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v32 / 3u + 1) | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1)) >> 2)
                 | (4 * v32 / 3u + 1)
                 | ((unsigned __int64)(4 * v32 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 632) = v36;
            v37 = (_DWORD *)sub_C7D670(72 * v36, 8);
            v38 = *(unsigned int *)(a1 + 632);
            *(_QWORD *)(a1 + 624) = 0;
            *(_QWORD *)(a1 + 616) = v37;
            for ( i = &v37[18 * v38]; i != v37; v37 += 18 )
            {
              if ( v37 )
              {
                *v37 = -1;
                v37[1] = 0x7FFFFFFF;
              }
            }
          }
          break;
        }
        if ( (_DWORD)v31 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 616), v17, 8);
          *(_QWORD *)(a1 + 616) = 0;
          *(_QWORD *)(a1 + 624) = 0;
          *(_DWORD *)(a1 + 632) = 0;
          break;
        }
LABEL_32:
        *(_QWORD *)(a1 + 624) = 0;
        break;
      }
    }
  }
  v20 = *(_DWORD *)(a1 + 480);
  ++*(_QWORD *)(a1 + 464);
  if ( !v20 )
  {
    if ( !*(_DWORD *)(a1 + 484) )
      goto LABEL_39;
    v21 = *(unsigned int *)(a1 + 488);
    if ( (unsigned int)v21 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 472), 16 * v21, 8);
      *(_QWORD *)(a1 + 472) = 0;
      *(_QWORD *)(a1 + 480) = 0;
      *(_DWORD *)(a1 + 488) = 0;
      goto LABEL_39;
    }
    goto LABEL_36;
  }
  v25 = 4 * v20;
  v21 = *(unsigned int *)(a1 + 488);
  if ( (unsigned int)(4 * v20) < 0x40 )
    v25 = 64;
  if ( v25 >= (unsigned int)v21 )
  {
LABEL_36:
    v22 = *(_QWORD **)(a1 + 472);
    for ( j = &v22[2 * v21]; j != v22; v22 += 2 )
      *v22 = -4096;
    *(_QWORD *)(a1 + 480) = 0;
    goto LABEL_39;
  }
  v26 = v20 - 1;
  if ( v26 )
  {
    _BitScanReverse(&v26, v26);
    v27 = *(_QWORD **)(a1 + 472);
    v28 = 1 << (33 - (v26 ^ 0x1F));
    if ( v28 < 64 )
      v28 = 64;
    if ( v28 == (_DWORD)v21 )
    {
      *(_QWORD *)(a1 + 480) = 0;
      v29 = &v27[2 * (unsigned int)v28];
      do
      {
        if ( v27 )
          *v27 = -4096;
        v27 += 2;
      }
      while ( v29 != v27 );
      goto LABEL_39;
    }
  }
  else
  {
    v27 = *(_QWORD **)(a1 + 472);
    v28 = 64;
  }
  sub_C7D6A0((__int64)v27, 16 * v21, 8);
  v40 = ((((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
       | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
       | (4 * v28 / 3u + 1)
       | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 16;
  v41 = (v40
       | (((((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
           | (4 * v28 / 3u + 1)
           | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
         | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
         | (4 * v28 / 3u + 1)
         | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 4)
       | (((4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1)) >> 2)
       | (4 * v28 / 3u + 1)
       | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 488) = v41;
  v42 = (_QWORD *)sub_C7D670(16 * v41, 8);
  v43 = *(unsigned int *)(a1 + 488);
  *(_QWORD *)(a1 + 480) = 0;
  *(_QWORD *)(a1 + 472) = v42;
  for ( k = &v42[2 * v43]; k != v42; v42 += 2 )
  {
    if ( v42 )
      *v42 = -4096;
  }
LABEL_39:
  result = *(_QWORD *)(a1 + 320);
  if ( result != *(_QWORD *)(a1 + 328) )
    *(_QWORD *)(a1 + 328) = result;
  return result;
}
