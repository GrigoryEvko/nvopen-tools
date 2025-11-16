// Function: sub_18CD460
// Address: 0x18cd460
//
__int64 __fastcall sub_18CD460(__int64 a1)
{
  int v2; // eax
  __int64 v3; // rdx
  _QWORD *v4; // rax
  _QWORD *j; // rdx
  int v6; // r14d
  __int64 result; // rax
  __int64 *v8; // rbx
  __int64 v9; // rdx
  __int64 *v10; // r13
  unsigned int v11; // ecx
  _QWORD *v12; // rdi
  unsigned int v13; // eax
  int v14; // eax
  unsigned __int64 v15; // rax
  __int64 v16; // rax
  int v17; // ebx
  __int64 v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // rdx
  _QWORD *i; // rdx
  int v22; // edx
  int v23; // ebx
  unsigned int v24; // r14d
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 k; // rdx
  _QWORD *v31; // rax

  v2 = *(_DWORD *)(a1 + 184);
  ++*(_QWORD *)(a1 + 168);
  if ( v2 )
  {
    v11 = 4 * v2;
    v3 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)(4 * v2) < 0x40 )
      v11 = 64;
    if ( (unsigned int)v3 <= v11 )
      goto LABEL_4;
    v12 = *(_QWORD **)(a1 + 176);
    v13 = v2 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v13, v13);
      v14 = 1 << (33 - (v13 ^ 0x1F));
      if ( v14 < 64 )
        v14 = 64;
      if ( (_DWORD)v3 == v14 )
      {
        *(_QWORD *)(a1 + 184) = 0;
        v31 = &v12[3 * v3];
        do
        {
          if ( v12 )
          {
            *v12 = -8;
            v12[1] = -8;
          }
          v12 += 3;
        }
        while ( v31 != v12 );
        goto LABEL_7;
      }
      v15 = (4 * v14 / 3u + 1) | ((unsigned __int64)(4 * v14 / 3u + 1) >> 1);
      v16 = ((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8)
           | (v15 >> 2)
           | v15
           | (((v15 >> 2) | v15) >> 4)
           | (((((v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 8) | (v15 >> 2) | v15 | (((v15 >> 2) | v15) >> 4)) >> 16))
          + 1;
      v17 = v16;
      v18 = 24 * v16;
    }
    else
    {
      v18 = 3072;
      v17 = 128;
    }
    j___libc_free_0(v12);
    *(_DWORD *)(a1 + 192) = v17;
    v19 = (_QWORD *)sub_22077B0(v18);
    v20 = *(unsigned int *)(a1 + 192);
    *(_QWORD *)(a1 + 184) = 0;
    *(_QWORD *)(a1 + 176) = v19;
    for ( i = &v19[3 * v20]; i != v19; v19 += 3 )
    {
      if ( v19 )
      {
        *v19 = -8;
        v19[1] = -8;
      }
    }
  }
  else if ( *(_DWORD *)(a1 + 188) )
  {
    v3 = *(unsigned int *)(a1 + 192);
    if ( (unsigned int)v3 <= 0x40 )
    {
LABEL_4:
      v4 = *(_QWORD **)(a1 + 176);
      for ( j = &v4[3 * v3]; j != v4; *(v4 - 2) = -8 )
      {
        *v4 = -8;
        v4 += 3;
      }
      *(_QWORD *)(a1 + 184) = 0;
      goto LABEL_7;
    }
    j___libc_free_0(*(_QWORD *)(a1 + 176));
    *(_QWORD *)(a1 + 176) = 0;
    *(_QWORD *)(a1 + 184) = 0;
    *(_DWORD *)(a1 + 192) = 0;
  }
LABEL_7:
  v6 = *(_DWORD *)(a1 + 216);
  ++*(_QWORD *)(a1 + 200);
  if ( !v6 )
  {
    result = *(unsigned int *)(a1 + 220);
    if ( !(_DWORD)result )
      return result;
  }
  v8 = *(__int64 **)(a1 + 208);
  result = (unsigned int)(4 * v6);
  v9 = *(unsigned int *)(a1 + 224);
  v10 = &v8[4 * v9];
  if ( (unsigned int)result < 0x40 )
    result = 64;
  if ( (unsigned int)v9 <= (unsigned int)result )
  {
    for ( ; v8 != v10; v8 += 4 )
    {
      result = *v8;
      if ( *v8 != -8 )
      {
        if ( result != -16 )
        {
          result = v8[3];
          if ( result != -8 && result != 0 && result != -16 )
            result = sub_1649B30(v8 + 1);
        }
        *v8 = -8;
      }
    }
    goto LABEL_20;
  }
  do
  {
    while ( 1 )
    {
      result = *v8;
      if ( *v8 != -8 )
        break;
LABEL_38:
      v8 += 4;
      if ( v8 == v10 )
        goto LABEL_42;
    }
    if ( result != -16 )
    {
      result = v8[3];
      if ( result != -8 && result != 0 && result != -16 )
        result = sub_1649B30(v8 + 1);
      goto LABEL_38;
    }
    v8 += 4;
  }
  while ( v8 != v10 );
LABEL_42:
  v22 = *(_DWORD *)(a1 + 224);
  if ( v6 )
  {
    v23 = 64;
    v24 = v6 - 1;
    if ( v24 )
    {
      _BitScanReverse(&v25, v24);
      v23 = 1 << (33 - (v25 ^ 0x1F));
      if ( v23 < 64 )
        v23 = 64;
    }
    v26 = *(_QWORD **)(a1 + 208);
    if ( v23 == v22 )
    {
      *(_QWORD *)(a1 + 216) = 0;
      result = (__int64)&v26[4 * (unsigned int)v23];
      do
      {
        if ( v26 )
          *v26 = -8;
        v26 += 4;
      }
      while ( (_QWORD *)result != v26 );
    }
    else
    {
      j___libc_free_0(v26);
      v27 = ((((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 16;
      v28 = (v27
           | (((((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
               | (4 * v23 / 3u + 1)
               | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
             | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
             | (4 * v23 / 3u + 1)
             | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 4)
           | (((4 * v23 / 3u + 1) | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1)) >> 2)
           | (4 * v23 / 3u + 1)
           | ((unsigned __int64)(4 * v23 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 224) = v28;
      result = sub_22077B0(32 * v28);
      v29 = *(unsigned int *)(a1 + 224);
      *(_QWORD *)(a1 + 216) = 0;
      *(_QWORD *)(a1 + 208) = result;
      for ( k = result + 32 * v29; k != result; result += 32 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
    }
  }
  else
  {
    if ( v22 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 208));
      *(_QWORD *)(a1 + 208) = 0;
      *(_QWORD *)(a1 + 216) = 0;
      *(_DWORD *)(a1 + 224) = 0;
      return result;
    }
LABEL_20:
    *(_QWORD *)(a1 + 216) = 0;
  }
  return result;
}
