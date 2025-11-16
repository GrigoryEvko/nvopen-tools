// Function: sub_2E596B0
// Address: 0x2e596b0
//
void __fastcall sub_2E596B0(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _DWORD *v4; // rax
  _DWORD *i; // rdx
  __int64 v6; // rax
  int v7; // r15d
  _QWORD *v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // r14
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned int v16; // ecx
  unsigned int v17; // eax
  _DWORD *v18; // rdi
  __int64 v19; // rbx
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // rax
  _DWORD *v22; // rax
  __int64 v23; // rdx
  _DWORD *j; // rdx
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r8
  unsigned __int64 v27; // rdi
  int v28; // edx
  int v29; // ebx
  unsigned int v30; // r15d
  unsigned int v31; // eax
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rdi
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *k; // rdx
  _DWORD *v38; // rax
  _QWORD *v39; // rax
  unsigned __int64 v40; // [rsp+8h] [rbp-38h]
  unsigned __int64 v41; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 72);
  ++*(_QWORD *)(a1 + 56);
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 76) )
      goto LABEL_7;
    v3 = *(unsigned int *)(a1 + 80);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 64), 8 * v3, 4);
      *(_QWORD *)(a1 + 64) = 0;
      *(_QWORD *)(a1 + 72) = 0;
      *(_DWORD *)(a1 + 80) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v16 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 80);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v16 = 64;
  if ( (unsigned int)v3 <= v16 )
  {
LABEL_4:
    v4 = *(_DWORD **)(a1 + 64);
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -1;
    *(_QWORD *)(a1 + 72) = 0;
    goto LABEL_7;
  }
  v17 = v2 - 1;
  if ( !v17 )
  {
    v18 = *(_DWORD **)(a1 + 64);
    LODWORD(v19) = 64;
LABEL_34:
    sub_C7D6A0((__int64)v18, 8 * v3, 4);
    v20 = ((((((((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v19 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v19 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v19 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v19 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 16;
    v21 = (v20
         | (((((((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v19 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v19 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v19 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v19 / 3u + 1) | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v19 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v19 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 80) = v21;
    v22 = (_DWORD *)sub_C7D670(8 * v21, 4);
    v23 = *(unsigned int *)(a1 + 80);
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 64) = v22;
    for ( j = &v22[2 * v23]; j != v22; v22 += 2 )
    {
      if ( v22 )
        *v22 = -1;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v17, v17);
  v18 = *(_DWORD **)(a1 + 64);
  v19 = (unsigned int)(1 << (33 - (v17 ^ 0x1F)));
  if ( (int)v19 < 64 )
    v19 = 64;
  if ( (_DWORD)v19 != (_DWORD)v3 )
    goto LABEL_34;
  *(_QWORD *)(a1 + 72) = 0;
  v38 = &v18[2 * v19];
  do
  {
    if ( v18 )
      *v18 = -1;
    v18 += 2;
  }
  while ( v38 != v18 );
LABEL_7:
  v6 = *(_QWORD *)(a1 + 88);
  if ( v6 != *(_QWORD *)(a1 + 96) )
    *(_QWORD *)(a1 + 96) = v6;
  v7 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  if ( v7 || *(_DWORD *)(a1 + 132) )
  {
    v8 = *(_QWORD **)(a1 + 120);
    v9 = 4 * v7;
    v10 = *(unsigned int *)(a1 + 136);
    v11 = 16 * v10;
    if ( (unsigned int)(4 * v7) < 0x40 )
      v9 = 64;
    v12 = &v8[(unsigned __int64)v11 / 8];
    if ( (unsigned int)v10 <= v9 )
    {
      for ( ; v8 != v12; v8 += 2 )
      {
        if ( *v8 != -4096 )
        {
          if ( *v8 != -8192 )
          {
            v13 = v8[1];
            if ( v13 )
            {
              v14 = *(_QWORD *)(v13 + 96);
              if ( v14 != v13 + 112 )
                _libc_free(v14);
              v15 = *(_QWORD *)(v13 + 24);
              if ( v15 != v13 + 40 )
                _libc_free(v15);
              j_j___libc_free_0(v13);
            }
          }
          *v8 = -4096;
        }
      }
      goto LABEL_25;
    }
    while ( 1 )
    {
      while ( *v8 == -8192 || *v8 == -4096 )
      {
LABEL_44:
        v8 += 2;
        if ( v8 == v12 )
          goto LABEL_49;
      }
      v26 = v8[1];
      if ( v26 )
      {
        v25 = *(_QWORD *)(v26 + 96);
        if ( v25 != v26 + 112 )
        {
          v40 = v8[1];
          _libc_free(v25);
          v26 = v40;
        }
        v27 = *(_QWORD *)(v26 + 24);
        if ( v27 != v26 + 40 )
        {
          v41 = v26;
          _libc_free(v27);
          v26 = v41;
        }
        j_j___libc_free_0(v26);
        goto LABEL_44;
      }
      v8 += 2;
      if ( v8 == v12 )
      {
LABEL_49:
        v28 = *(_DWORD *)(a1 + 136);
        if ( v7 )
        {
          v29 = 64;
          v30 = v7 - 1;
          if ( v30 )
          {
            _BitScanReverse(&v31, v30);
            v29 = 1 << (33 - (v31 ^ 0x1F));
            if ( v29 < 64 )
              v29 = 64;
          }
          v32 = *(_QWORD **)(a1 + 120);
          if ( v29 == v28 )
          {
            *(_QWORD *)(a1 + 128) = 0;
            v39 = &v32[2 * (unsigned int)v29];
            do
            {
              if ( v32 )
                *v32 = -4096;
              v32 += 2;
            }
            while ( v39 != v32 );
          }
          else
          {
            sub_C7D6A0((__int64)v32, v11, 8);
            v33 = ((((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                     | (4 * v29 / 3u + 1)
                     | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                   | (4 * v29 / 3u + 1)
                   | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                   | (4 * v29 / 3u + 1)
                   | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                 | (4 * v29 / 3u + 1)
                 | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 16;
            v34 = (v33
                 | (((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                     | (4 * v29 / 3u + 1)
                     | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                   | (4 * v29 / 3u + 1)
                   | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                   | (4 * v29 / 3u + 1)
                   | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
                 | (4 * v29 / 3u + 1)
                 | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 136) = v34;
            v35 = (_QWORD *)sub_C7D670(16 * v34, 8);
            v36 = *(unsigned int *)(a1 + 136);
            *(_QWORD *)(a1 + 128) = 0;
            *(_QWORD *)(a1 + 120) = v35;
            for ( k = &v35[2 * v36]; k != v35; v35 += 2 )
            {
              if ( v35 )
                *v35 = -4096;
            }
          }
          break;
        }
        if ( v28 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 120), v11, 8);
          *(_QWORD *)(a1 + 120) = 0;
          *(_QWORD *)(a1 + 128) = 0;
          *(_DWORD *)(a1 + 136) = 0;
          break;
        }
LABEL_25:
        *(_QWORD *)(a1 + 128) = 0;
        break;
      }
    }
  }
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  sub_307CD20(a1);
  sub_2E592C0((__int64 *)a1);
}
