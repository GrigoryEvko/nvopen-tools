// Function: sub_3741110
// Address: 0x3741110
//
__int64 __fastcall sub_3741110(__int64 a1)
{
  __int64 v2; // rdi
  __int64 *v3; // r13
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // r14
  __int64 *v7; // rax
  __int64 *v8; // rsi
  __int64 v9; // rax
  __int64 *v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // r8
  int v16; // eax
  __int64 v17; // r9
  int v18; // edx
  unsigned int v19; // eax
  int v20; // r10d
  int v21; // r11d
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 *v25; // rbx
  int v26; // eax
  __int64 v27; // rdx
  _QWORD *v28; // rax
  _QWORD *i; // rdx
  __int64 result; // rax
  unsigned int v31; // ecx
  unsigned int v32; // eax
  _QWORD *v33; // rdi
  int v34; // ebx
  _QWORD *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rsi
  __int64 *v40; // r14
  unsigned __int8 *v41; // rsi
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // rax
  _QWORD *v44; // rax
  __int64 v45; // rdx
  _QWORD *j; // rdx
  __int64 v47; // rsi
  __int64 *v48; // rax
  __int64 v49[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(a1 + 160);
  v3 = *(__int64 **)(a1 + 168);
  if ( (__int64 *)v2 == v3 )
    goto LABEL_38;
  if ( !v2 )
    BUG();
  v4 = *(_QWORD *)v2;
  v5 = v2;
  if ( (*(_QWORD *)v2 & 4) == 0 && (*(_BYTE *)(v2 + 44) & 8) != 0 )
  {
    do
      v5 = *(_QWORD *)(v5 + 8);
    while ( (*(_BYTE *)(v5 + 44) & 8) != 0 );
  }
  v6 = *(__int64 **)(v5 + 8);
  if ( v3 || (v36 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL), v3 = (__int64 *)(v36 + 48), v2 != v36 + 48) )
  {
    while ( 1 )
    {
      v7 = (__int64 *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
      v8 = v7;
      if ( !v7 )
        BUG();
      v9 = *v7;
      v10 = v8;
      if ( (v9 & 4) == 0 && (*((_BYTE *)v8 + 44) & 4) != 0 )
      {
        while ( 1 )
        {
          v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
          v10 = (__int64 *)v11;
          if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
            break;
          v9 = *(_QWORD *)v11;
        }
      }
      v12 = *(_QWORD *)(v2 + 32);
      v13 = 0;
      v14 = v12 + 40LL * (*(_DWORD *)(v2 + 40) & 0xFFFFFF);
      if ( v12 != v14 )
      {
        do
        {
          if ( !*(_BYTE *)v12 )
          {
            if ( (*(_BYTE *)(v12 + 3) & 0x10) != 0 )
            {
              if ( (_DWORD)v13 )
                goto LABEL_21;
              v13 = *(unsigned int *)(v12 + 8);
            }
            else if ( *(int *)(v12 + 8) < 0 )
            {
              goto LABEL_21;
            }
          }
          v12 += 40;
        }
        while ( v14 != v12 );
        if ( (_DWORD)v13 )
        {
          v15 = *(_QWORD *)(a1 + 40);
          v16 = *(_DWORD *)(v15 + 520);
          v17 = *(_QWORD *)(v15 + 504);
          if ( v16 )
          {
            v18 = v16 - 1;
            v19 = (v16 - 1) & (37 * v13);
            v20 = *(_DWORD *)(v17 + 4LL * v19);
            if ( v20 == (_DWORD)v13 )
              goto LABEL_21;
            v21 = 1;
            while ( v20 != -1 )
            {
              v19 = v18 & (v21 + v19);
              v20 = *(_DWORD *)(v17 + 4LL * v19);
              if ( (_DWORD)v13 == v20 )
                goto LABEL_21;
              ++v21;
            }
          }
          v22 = *(_QWORD *)(v15 + 856);
          v23 = *(_QWORD *)(v15 + 864);
          if ( v22 == v23 )
          {
LABEL_62:
            v37 = *(_QWORD *)(a1 + 56);
            if ( (int)v13 < 0 )
              v38 = *(_QWORD *)(*(_QWORD *)(v37 + 56) + 16 * (v13 & 0x7FFFFFFF) + 8);
            else
              v38 = *(_QWORD *)(*(_QWORD *)(v37 + 304) + 8 * v13);
            while ( v38 )
            {
              if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 && (*(_BYTE *)(v38 + 4) & 8) == 0 )
                goto LABEL_21;
              v38 = *(_QWORD *)(v38 + 32);
            }
            if ( *(_QWORD *)(a1 + 168) == v2 )
            {
              v48 = 0;
              if ( v2 != *(_QWORD *)(*(_QWORD *)(v2 + 24) + 56LL) )
                v48 = v8;
              *(_QWORD *)(a1 + 168) = v48;
            }
            sub_2E88E20(v2);
          }
          else
          {
            while ( *(_DWORD *)(v22 + 8) != (_DWORD)v13 )
            {
              v22 += 16;
              if ( v23 == v22 )
                goto LABEL_62;
            }
          }
        }
      }
LABEL_21:
      if ( v3 == v10 )
      {
        v36 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 744LL);
        v3 = (__int64 *)(v36 + 48);
        break;
      }
      v4 = *v10;
      v2 = (__int64)v10;
    }
  }
  if ( v6 != v3 )
  {
    v24 = *(_QWORD *)(a1 + 168);
    if ( v24 )
    {
      if ( (*(_BYTE *)v24 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v24 + 44) & 8) != 0 )
          v24 = *(_QWORD *)(v24 + 8);
      }
      v25 = *(__int64 **)(v24 + 8);
    }
    else
    {
      v25 = *(__int64 **)(v36 + 56);
    }
    if ( v25 != v6 && !v25[7] )
    {
      v39 = v6[7];
      v40 = v25 + 7;
      v49[0] = v39;
      if ( v39 )
      {
        sub_B96E90((__int64)v49, v39, 1);
        if ( v40 == v49 )
        {
          if ( v49[0] )
            sub_B91220((__int64)(v25 + 7), v49[0]);
          goto LABEL_38;
        }
        v47 = v25[7];
        if ( v47 )
          sub_B91220((__int64)(v25 + 7), v47);
      }
      else if ( v40 == v49 )
      {
        goto LABEL_38;
      }
      v41 = (unsigned __int8 *)v49[0];
      v25[7] = v49[0];
      if ( v41 )
        sub_B976B0((__int64)v49, v41, (__int64)(v25 + 7));
    }
  }
LABEL_38:
  v26 = *(_DWORD *)(a1 + 24);
  ++*(_QWORD *)(a1 + 8);
  if ( !v26 )
  {
    if ( !*(_DWORD *)(a1 + 28) )
      goto LABEL_44;
    v27 = *(unsigned int *)(a1 + 32);
    if ( (unsigned int)v27 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 16), 16LL * *(unsigned int *)(a1 + 32), 8);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      goto LABEL_44;
    }
    goto LABEL_41;
  }
  v31 = 4 * v26;
  v27 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)(4 * v26) < 0x40 )
    v31 = 64;
  if ( v31 >= (unsigned int)v27 )
  {
LABEL_41:
    v28 = *(_QWORD **)(a1 + 16);
    for ( i = &v28[2 * v27]; i != v28; v28 += 2 )
      *v28 = -4096;
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_44;
  }
  v32 = v26 - 1;
  if ( v32 )
  {
    _BitScanReverse(&v32, v32);
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 1 << (33 - (v32 ^ 0x1F));
    if ( v34 < 64 )
      v34 = 64;
    if ( (_DWORD)v27 == v34 )
    {
      *(_QWORD *)(a1 + 24) = 0;
      v35 = &v33[2 * (unsigned int)v27];
      do
      {
        if ( v33 )
          *v33 = -4096;
        v33 += 2;
      }
      while ( v35 != v33 );
      goto LABEL_44;
    }
  }
  else
  {
    v33 = *(_QWORD **)(a1 + 16);
    v34 = 64;
  }
  sub_C7D6A0((__int64)v33, 16LL * *(unsigned int *)(a1 + 32), 8);
  v42 = ((((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
       | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
       | (4 * v34 / 3u + 1)
       | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 16;
  v43 = (v42
       | (((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
           | (4 * v34 / 3u + 1)
           | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
         | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
         | (4 * v34 / 3u + 1)
         | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 4)
       | (((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
       | (4 * v34 / 3u + 1)
       | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 32) = v43;
  v44 = (_QWORD *)sub_C7D670(16 * v43, 8);
  v45 = *(unsigned int *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 16) = v44;
  for ( j = &v44[2 * v45]; j != v44; v44 += 2 )
  {
    if ( v44 )
      *v44 = -4096;
  }
LABEL_44:
  *(_QWORD *)(a1 + 160) = *(_QWORD *)(a1 + 168);
  sub_3741080(a1);
  result = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 752LL);
  *(_QWORD *)(a1 + 176) = result;
  return result;
}
