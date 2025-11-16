// Function: sub_B80C30
// Address: 0xb80c30
//
__int64 __fastcall sub_B80C30(__int64 a1)
{
  __int64 *v2; // r13
  __int64 *i; // r14
  __int64 v4; // r12
  int v5; // eax
  __int64 v6; // rdx
  _QWORD *v7; // rax
  _QWORD *j; // rdx
  __int64 *v9; // r13
  __int64 result; // rax
  __int64 *m; // r14
  __int64 v12; // r12
  int v13; // eax
  __int64 v14; // rdx
  __int64 n; // rdx
  unsigned int v16; // ecx
  unsigned int v17; // eax
  _QWORD *v18; // rdi
  int v19; // r15d
  unsigned int v20; // ecx
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rdi
  __int64 v25; // rdi
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *k; // rdx
  unsigned __int64 v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 ii; // rdx
  _QWORD *v33; // rax
  int v34; // [rsp+Ch] [rbp-34h]

  v2 = *(__int64 **)(a1 + 32);
  for ( i = &v2[*(unsigned int *)(a1 + 40)]; i != v2; *(_OWORD *)(v4 + 192) = 0 )
  {
    v4 = *v2;
    v5 = *(_DWORD *)(*v2 + 224);
    ++*(_QWORD *)(*v2 + 208);
    if ( v5 )
    {
      v20 = 4 * v5;
      v6 = *(unsigned int *)(v4 + 232);
      if ( (unsigned int)(4 * v5) < 0x40 )
        v20 = 64;
      if ( v20 >= (unsigned int)v6 )
      {
LABEL_5:
        v7 = *(_QWORD **)(v4 + 216);
        for ( j = &v7[2 * v6]; j != v7; v7 += 2 )
          *v7 = -4096;
        *(_QWORD *)(v4 + 224) = 0;
        goto LABEL_8;
      }
      v21 = v5 - 1;
      if ( v21 )
      {
        _BitScanReverse(&v21, v21);
        v22 = *(_QWORD **)(v4 + 216);
        v23 = (unsigned int)(1 << (33 - (v21 ^ 0x1F)));
        if ( (int)v23 < 64 )
          v23 = 64;
        if ( (_DWORD)v23 == (_DWORD)v6 )
        {
          *(_QWORD *)(v4 + 224) = 0;
          v33 = &v22[2 * v23];
          do
          {
            if ( v22 )
              *v22 = -4096;
            v22 += 2;
          }
          while ( v33 != v22 );
          goto LABEL_8;
        }
      }
      else
      {
        v22 = *(_QWORD **)(v4 + 216);
        LODWORD(v23) = 64;
      }
      v34 = v23;
      sub_C7D6A0(v22, 16LL * (unsigned int)v6, 8);
      v24 = (((((((4 * v34 / 3u + 1) | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1)) >> 2)
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
          | ((unsigned __int64)(4 * v34 / 3u + 1) >> 1);
      v25 = ((v24 >> 16) | v24) + 1;
      *(_DWORD *)(v4 + 232) = v25;
      v26 = (_QWORD *)sub_C7D670(16 * v25, 8);
      v27 = *(unsigned int *)(v4 + 232);
      *(_QWORD *)(v4 + 224) = 0;
      *(_QWORD *)(v4 + 216) = v26;
      for ( k = &v26[2 * v27]; k != v26; v26 += 2 )
      {
        if ( v26 )
          *v26 = -4096;
      }
    }
    else if ( *(_DWORD *)(v4 + 228) )
    {
      v6 = *(unsigned int *)(v4 + 232);
      if ( (unsigned int)v6 <= 0x40 )
        goto LABEL_5;
      sub_C7D6A0(*(_QWORD *)(v4 + 216), 16LL * (unsigned int)v6, 8);
      *(_QWORD *)(v4 + 216) = 0;
      *(_QWORD *)(v4 + 224) = 0;
      *(_DWORD *)(v4 + 232) = 0;
    }
LABEL_8:
    ++v2;
    *(_OWORD *)(v4 + 160) = 0;
    *(_OWORD *)(v4 + 176) = 0;
  }
  v9 = *(__int64 **)(a1 + 112);
  result = *(unsigned int *)(a1 + 120);
  for ( m = &v9[result]; m != v9; *(_OWORD *)(v12 + 192) = 0 )
  {
    v12 = *v9;
    v13 = *(_DWORD *)(*v9 + 224);
    ++*(_QWORD *)(*v9 + 208);
    if ( v13 )
    {
      v16 = 4 * v13;
      v14 = *(unsigned int *)(v12 + 232);
      if ( (unsigned int)(4 * v13) < 0x40 )
        v16 = 64;
      if ( (unsigned int)v14 <= v16 )
      {
LABEL_13:
        result = *(_QWORD *)(v12 + 216);
        for ( n = result + 16 * v14; n != result; result += 16 )
          *(_QWORD *)result = -4096;
        *(_QWORD *)(v12 + 224) = 0;
        goto LABEL_16;
      }
      v17 = v13 - 1;
      if ( !v17 )
      {
        v18 = *(_QWORD **)(v12 + 216);
        v19 = 64;
LABEL_43:
        sub_C7D6A0(v18, 16LL * (unsigned int)v14, 8);
        v29 = (((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
                | (4 * v19 / 3u + 1)
                | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
              | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
              | (4 * v19 / 3u + 1)
              | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
            | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
              | (4 * v19 / 3u + 1)
              | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
            | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
            | (4 * v19 / 3u + 1)
            | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1);
        v30 = ((v29 >> 16) | v29) + 1;
        *(_DWORD *)(v12 + 232) = v30;
        result = sub_C7D670(16 * v30, 8);
        v31 = *(unsigned int *)(v12 + 232);
        *(_QWORD *)(v12 + 224) = 0;
        *(_QWORD *)(v12 + 216) = result;
        for ( ii = result + 16 * v31; ii != result; result += 16 )
        {
          if ( result )
            *(_QWORD *)result = -4096;
        }
        goto LABEL_16;
      }
      _BitScanReverse(&v17, v17);
      v18 = *(_QWORD **)(v12 + 216);
      v19 = 1 << (33 - (v17 ^ 0x1F));
      if ( v19 < 64 )
        v19 = 64;
      if ( v19 != (_DWORD)v14 )
        goto LABEL_43;
      *(_QWORD *)(v12 + 224) = 0;
      result = (__int64)&v18[2 * (unsigned int)v19];
      do
      {
        if ( v18 )
          *v18 = -4096;
        v18 += 2;
      }
      while ( (_QWORD *)result != v18 );
    }
    else
    {
      result = *(unsigned int *)(v12 + 228);
      if ( (_DWORD)result )
      {
        v14 = *(unsigned int *)(v12 + 232);
        if ( (unsigned int)v14 <= 0x40 )
          goto LABEL_13;
        result = sub_C7D6A0(*(_QWORD *)(v12 + 216), 16LL * (unsigned int)v14, 8);
        *(_QWORD *)(v12 + 216) = 0;
        *(_QWORD *)(v12 + 224) = 0;
        *(_DWORD *)(v12 + 232) = 0;
      }
    }
LABEL_16:
    ++v9;
    *(_OWORD *)(v12 + 160) = 0;
    *(_OWORD *)(v12 + 176) = 0;
  }
  return result;
}
