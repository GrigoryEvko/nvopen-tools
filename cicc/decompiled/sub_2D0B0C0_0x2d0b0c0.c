// Function: sub_2D0B0C0
// Address: 0x2d0b0c0
//
_BYTE *__fastcall sub_2D0B0C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // eax
  __int64 i; // rdx
  _QWORD *v7; // rax
  __int64 v8; // rax
  int v9; // r15d
  _QWORD *v10; // rbx
  unsigned int v11; // eax
  __int64 v12; // r13
  _QWORD *v13; // r14
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rdi
  int v20; // eax
  int v21; // ebx
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r8
  unsigned __int64 v25; // rdi
  int v26; // ebx
  unsigned int v27; // r15d
  unsigned int v28; // eax
  _QWORD *v29; // rdi
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rax
  unsigned __int64 v33; // rdi
  _QWORD *v34; // rax
  unsigned __int64 v35; // [rsp+8h] [rbp-38h]
  unsigned __int64 v36; // [rsp+8h] [rbp-38h]

  v5 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v5 )
  {
    i = *(unsigned int *)(a1 + 76);
    if ( !(_DWORD)i )
      goto LABEL_7;
    i = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)i > 0x40 )
    {
      a2 = 16LL * (unsigned int)i;
      sub_C7D6A0(*(_QWORD *)(a1 + 64), a2, 8);
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  a4 = (unsigned int)(4 * v5);
  a2 = 64;
  i = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)a4 < 0x40 )
    a4 = 64;
  if ( (unsigned int)i <= (unsigned int)a4 )
  {
LABEL_4:
    v7 = *(_QWORD **)(a1 + 64);
    for ( i = (__int64)&v7[2 * i]; (_QWORD *)i != v7; v7 += 2 )
      *v7 = -4096;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_7;
  }
  v18 = v5 - 1;
  if ( !v18 )
  {
    v19 = *(_QWORD **)(a1 + 64);
    v21 = 64;
LABEL_60:
    sub_C7D6A0((__int64)v19, 16LL * (unsigned int)i, 8);
    a2 = 8;
    v32 = ((((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 16;
    v33 = (v32
         | (((((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
             | (4 * v21 / 3u + 1)
             | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
           | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
           | (4 * v21 / 3u + 1)
           | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 4)
         | (((4 * v21 / 3u + 1) | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1)) >> 2)
         | (4 * v21 / 3u + 1)
         | ((unsigned __int64)(4 * v21 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 80) = v33;
    *(_QWORD *)(a1 + 64) = sub_C7D670(16 * v33, 8);
    sub_2D06D20(a1 + 56);
    goto LABEL_7;
  }
  _BitScanReverse(&v18, v18);
  v19 = *(_QWORD **)(a1 + 64);
  v20 = v18 ^ 0x1F;
  a4 = (unsigned int)(33 - v20);
  v21 = 1 << (33 - v20);
  if ( v21 < 64 )
    v21 = 64;
  if ( (_DWORD)i != v21 )
    goto LABEL_60;
  *(_QWORD *)(a1 + 72) = 0;
  v22 = &v19[2 * (unsigned int)i];
  do
  {
    if ( v19 )
      *v19 = -4096;
    v19 += 2;
  }
  while ( v22 != v19 );
LABEL_7:
  v8 = *(_QWORD *)(a1 + 88);
  if ( v8 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v8;
  v9 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v9 || *(_DWORD *)(a1 + 132) )
  {
    a4 = 64;
    v10 = *(_QWORD **)(a1 + 120);
    v11 = 4 * v9;
    i = *(unsigned int *)(a1 + 136);
    v12 = 16 * i;
    if ( (unsigned int)(4 * v9) < 0x40 )
      v11 = 64;
    v13 = &v10[(unsigned __int64)v12 / 8];
    if ( (unsigned int)i <= v11 )
    {
      while ( v10 != v13 )
      {
        if ( *v10 != -4096 )
        {
          if ( *v10 != -8192 )
          {
            v14 = v10[1];
            if ( v14 )
            {
              v15 = *(_QWORD *)(v14 + 96);
              if ( v15 != v14 + 112 )
                _libc_free(v15);
              v16 = *(_QWORD *)(v14 + 24);
              if ( v16 != v14 + 40 )
                _libc_free(v16);
              a2 = 168;
              j_j___libc_free_0(v14);
            }
          }
          *v10 = -4096;
        }
        v10 += 2;
      }
      goto LABEL_26;
    }
    while ( 1 )
    {
      while ( *v10 == -8192 || *v10 == -4096 )
      {
LABEL_45:
        v10 += 2;
        if ( v10 == v13 )
          goto LABEL_50;
      }
      v24 = v10[1];
      if ( v24 )
      {
        v23 = *(_QWORD *)(v24 + 96);
        if ( v23 != v24 + 112 )
        {
          v35 = v10[1];
          _libc_free(v23);
          v24 = v35;
        }
        v25 = *(_QWORD *)(v24 + 24);
        if ( v25 != v24 + 40 )
        {
          v36 = v24;
          _libc_free(v25);
          v24 = v36;
        }
        a2 = 168;
        j_j___libc_free_0(v24);
        goto LABEL_45;
      }
      v10 += 2;
      if ( v10 == v13 )
      {
LABEL_50:
        i = *(unsigned int *)(a1 + 136);
        if ( v9 )
        {
          v26 = 64;
          v27 = v9 - 1;
          if ( v27 )
          {
            _BitScanReverse(&v28, v27);
            a4 = 33 - (v28 ^ 0x1F);
            v26 = 1 << (33 - (v28 ^ 0x1F));
            if ( v26 < 64 )
              v26 = 64;
          }
          v29 = *(_QWORD **)(a1 + 120);
          if ( v26 == (_DWORD)i )
          {
            *(_QWORD *)(a1 + 128) = 0;
            v34 = &v29[2 * (unsigned int)v26];
            do
            {
              if ( v29 )
                *v29 = -4096;
              v29 += 2;
            }
            while ( v34 != v29 );
          }
          else
          {
            sub_C7D6A0((__int64)v29, v12, 8);
            a2 = 8;
            v30 = ((((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                     | (4 * v26 / 3u + 1)
                     | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                   | (4 * v26 / 3u + 1)
                   | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                   | (4 * v26 / 3u + 1)
                   | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                 | (4 * v26 / 3u + 1)
                 | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 16;
            v31 = (v30
                 | (((((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                     | (4 * v26 / 3u + 1)
                     | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                   | (4 * v26 / 3u + 1)
                   | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                   | (4 * v26 / 3u + 1)
                   | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v26 / 3u + 1) | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1)) >> 2)
                 | (4 * v26 / 3u + 1)
                 | ((unsigned __int64)(4 * v26 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 136) = v31;
            *(_QWORD *)(a1 + 120) = sub_C7D670(16 * v31, 8);
            sub_2D06D60(a1 + 112);
          }
          break;
        }
        if ( (_DWORD)i )
        {
          a2 = v12;
          sub_C7D6A0(*(_QWORD *)(a1 + 120), v12, 8);
          *(_QWORD *)(a1 + 120) = 0;
          *(_QWORD *)(a1 + 128) = 0;
          *(_DWORD *)(a1 + 136) = 0;
          break;
        }
LABEL_26:
        *(_QWORD *)(a1 + 128) = 0;
        break;
      }
    }
  }
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_FCF420(a1, a2, i, a4);
  return sub_CE6110((_QWORD *)a1);
}
