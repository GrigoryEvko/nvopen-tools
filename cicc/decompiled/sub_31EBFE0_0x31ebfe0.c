// Function: sub_31EBFE0
// Address: 0x31ebfe0
//
void __fastcall sub_31EBFE0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r12
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rdx
  unsigned __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rsi
  int v10; // edi
  __int64 v11; // r9
  __int64 v12; // r8
  unsigned __int64 v13; // rdx
  const __m128i *v14; // r14
  __m128i *v15; // rax
  int v16; // r14d
  _QWORD *v17; // rbx
  unsigned int v18; // eax
  __int64 v19; // r15
  _QWORD *v20; // r12
  unsigned __int64 v21; // rdi
  __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // r12
  __int64 *v26; // rbx
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rax
  int v30; // ebx
  unsigned int v31; // edx
  _QWORD *v32; // rdi
  unsigned __int64 v33; // rdx
  unsigned __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *i; // rdx
  __int64 v38; // rdi
  __int64 v39; // r8
  char *v40; // r14
  _QWORD *v41; // rax
  int v42; // [rsp+Ch] [rbp-A4h] BYREF
  __int128 v43; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v44; // [rsp+20h] [rbp-90h]
  __int64 *v45; // [rsp+28h] [rbp-88h]
  __int64 v46; // [rsp+30h] [rbp-80h] BYREF
  __int64 v47; // [rsp+38h] [rbp-78h]
  __int64 v48; // [rsp+40h] [rbp-70h]
  int v49; // [rsp+48h] [rbp-68h]
  __int64 v50[12]; // [rsp+50h] [rbp-60h] BYREF

  v3 = *a2;
  if ( *(_DWORD *)(a1 + 520) || (*(_BYTE *)(v3 + 7) & 0x20) != 0 && sub_B91C10(v3, 37) )
  {
    v4 = 4;
    if ( (unsigned int)(*(_DWORD *)(a2[1] + 636) - 3) <= 1 )
      v4 = sub_31DAFE0(a1);
    v50[1] = (__int64)a2;
    v45 = a2;
    v5 = *(_QWORD *)(a1 + 224);
    v42 = v4;
    v6 = *(unsigned int *)(v5 + 128);
    v50[0] = (__int64)&v43;
    v50[3] = (__int64)&v42;
    v7 = *(_QWORD *)(v5 + 120);
    v44 = a1;
    v8 = v6;
    v50[2] = a1;
    v9 = 32 * v6;
    v50[4] = v3;
    v43 = 0;
    if ( (_DWORD)v6 )
    {
      v22 = v7 + v9 - 32;
      v12 = *(_QWORD *)(v22 + 16);
      v8 = *(_DWORD *)(v22 + 24);
      v11 = *(_QWORD *)v22;
      v10 = *(_DWORD *)(v22 + 8);
    }
    else
    {
      v10 = 0;
      v11 = 0;
      v12 = 0;
    }
    v46 = v11;
    v13 = v6 + 1;
    LODWORD(v47) = v10;
    v14 = (const __m128i *)&v46;
    v48 = v12;
    v49 = v8;
    if ( v13 > *(unsigned int *)(v5 + 132) )
    {
      v38 = v5 + 120;
      v39 = v5 + 136;
      if ( v7 > (unsigned __int64)&v46 || (unsigned __int64)&v46 >= v7 + v9 )
      {
        v14 = (const __m128i *)&v46;
        sub_C8D5F0(v38, (const void *)(v5 + 136), v13, 0x20u, v39, v11);
        v7 = *(_QWORD *)(v5 + 120);
        v9 = 32LL * *(unsigned int *)(v5 + 128);
      }
      else
      {
        v40 = (char *)&v46 - v7;
        sub_C8D5F0(v38, (const void *)(v5 + 136), v13, 0x20u, v39, v11);
        v7 = *(_QWORD *)(v5 + 120);
        v14 = (const __m128i *)&v40[v7];
        v9 = 32LL * *(unsigned int *)(v5 + 128);
      }
    }
    v15 = (__m128i *)(v9 + v7);
    *v15 = _mm_loadu_si128(v14);
    v15[1] = _mm_loadu_si128(v14 + 1);
    ++*(_DWORD *)(v5 + 128);
    if ( (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
    {
      v9 = sub_B91C10(v3, 37);
      if ( v9 )
      {
        v46 = *(_QWORD *)(a1 + 536);
        v47 = *(_QWORD *)(a1 + 400);
        sub_31EBBA0(v50, v9, &v46, 2, 1u);
      }
    }
    if ( *(_DWORD *)(a1 + 520) )
    {
      v24 = *(__int64 **)(a1 + 512);
      v25 = &v24[9 * *(unsigned int *)(a1 + 528)];
      if ( v24 != v25 )
      {
        while ( 1 )
        {
          v9 = *v24;
          v26 = v24;
          if ( *v24 != -4096 && v9 != -8192 )
            break;
          v24 += 9;
          if ( v25 == v24 )
            goto LABEL_11;
        }
        if ( v24 != v25 )
        {
          while ( 1 )
          {
            v27 = (__int64 *)v26[1];
            v28 = *((unsigned int *)v26 + 4);
            v26 += 9;
            sub_31EBBA0(v50, v9, v27, v28, 0);
            if ( v26 == v25 )
              break;
            while ( *v26 == -8192 || *v26 == -4096 )
            {
              v26 += 9;
              if ( v25 == v26 )
                goto LABEL_11;
            }
            if ( v25 == v26 )
              break;
            v9 = *v26;
          }
        }
      }
    }
LABEL_11:
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 224) + 168LL))(*(_QWORD *)(a1 + 224), v9);
    v16 = *(_DWORD *)(a1 + 520);
    ++*(_QWORD *)(a1 + 504);
    if ( v16 || *(_DWORD *)(a1 + 524) )
    {
      v17 = *(_QWORD **)(a1 + 512);
      v18 = 4 * v16;
      v19 = 72LL * *(unsigned int *)(a1 + 528);
      if ( (unsigned int)(4 * v16) < 0x40 )
        v18 = 64;
      v20 = &v17[(unsigned __int64)v19 / 8];
      if ( *(_DWORD *)(a1 + 528) <= v18 )
      {
        while ( v17 != v20 )
        {
          if ( *v17 != -4096 )
          {
            if ( *v17 != -8192 )
            {
              v21 = v17[1];
              if ( (_QWORD *)v21 != v17 + 3 )
                _libc_free(v21);
            }
            *v17 = -4096;
          }
          v17 += 9;
        }
      }
      else
      {
        do
        {
          if ( *v17 != -8192 && *v17 != -4096 )
          {
            v23 = v17[1];
            if ( (_QWORD *)v23 != v17 + 3 )
              _libc_free(v23);
          }
          v17 += 9;
        }
        while ( v17 != v20 );
        v29 = *(unsigned int *)(a1 + 528);
        if ( v16 )
        {
          v30 = 64;
          if ( v16 != 1 )
          {
            _BitScanReverse(&v31, v16 - 1);
            v30 = 1 << (33 - (v31 ^ 0x1F));
            if ( v30 < 64 )
              v30 = 64;
          }
          v32 = *(_QWORD **)(a1 + 512);
          if ( (_DWORD)v29 == v30 )
          {
            *(_QWORD *)(a1 + 520) = 0;
            v41 = &v32[9 * v29];
            do
            {
              if ( v32 )
                *v32 = -4096;
              v32 += 9;
            }
            while ( v41 != v32 );
          }
          else
          {
            sub_C7D6A0((__int64)v32, v19, 8);
            v33 = ((((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                     | (4 * v30 / 3u + 1)
                     | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                   | (4 * v30 / 3u + 1)
                   | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                   | (4 * v30 / 3u + 1)
                   | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                 | (4 * v30 / 3u + 1)
                 | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 16;
            v34 = (v33
                 | (((((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                     | (4 * v30 / 3u + 1)
                     | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                   | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                   | (4 * v30 / 3u + 1)
                   | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 8)
                 | (((((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                   | (4 * v30 / 3u + 1)
                   | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v30 / 3u + 1) | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1)) >> 2)
                 | (4 * v30 / 3u + 1)
                 | ((unsigned __int64)(4 * v30 / 3u + 1) >> 1))
                + 1;
            *(_DWORD *)(a1 + 528) = v34;
            v35 = (_QWORD *)sub_C7D670(72 * v34, 8);
            v36 = *(unsigned int *)(a1 + 528);
            *(_QWORD *)(a1 + 520) = 0;
            *(_QWORD *)(a1 + 512) = v35;
            for ( i = &v35[9 * v36]; i != v35; v35 += 9 )
            {
              if ( v35 )
                *v35 = -4096;
            }
          }
          return;
        }
        if ( (_DWORD)v29 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 512), v19, 8);
          *(_QWORD *)(a1 + 512) = 0;
          *(_QWORD *)(a1 + 520) = 0;
          *(_DWORD *)(a1 + 528) = 0;
          return;
        }
      }
      *(_QWORD *)(a1 + 520) = 0;
    }
  }
}
