// Function: sub_196A390
// Address: 0x196a390
//
__int64 __fastcall sub_196A390(__int64 a1)
{
  int v2; // r14d
  _QWORD *v3; // rbx
  __int64 v4; // rdx
  _QWORD *v5; // r13
  unsigned int v6; // eax
  __int64 v7; // rax
  int v8; // eax
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 j; // rdx
  unsigned int v12; // ecx
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  int v15; // eax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  int v18; // ebx
  __int64 v19; // r13
  __int64 v20; // rdx
  __int64 k; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  int v24; // ebx
  unsigned int v25; // r14d
  unsigned int v26; // eax
  _QWORD *v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *i; // rdx
  _QWORD *v33; // rax

  v2 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( !v2 && !*(_DWORD *)(a1 + 44) )
    goto LABEL_18;
  v3 = *(_QWORD **)(a1 + 32);
  v4 = *(unsigned int *)(a1 + 48);
  v5 = &v3[5 * v4];
  v6 = 4 * v2;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  if ( (unsigned int)v4 <= v6 )
  {
    while ( 1 )
    {
      if ( v3 == v5 )
        goto LABEL_17;
      if ( *v3 != -8 )
        break;
      if ( v3[1] != -8 )
        goto LABEL_8;
LABEL_12:
      v3 += 5;
    }
    if ( *v3 != -16 || v3[1] != -16 )
    {
LABEL_8:
      v7 = v3[4];
      if ( v7 != 0 && v7 != -8 && v7 != -16 )
        sub_1649B30(v3 + 2);
    }
    *v3 = -8;
    v3[1] = -8;
    goto LABEL_12;
  }
  do
  {
    if ( *v3 == -8 )
    {
      if ( v3[1] == -8 )
        goto LABEL_44;
    }
    else if ( *v3 == -16 && v3[1] == -16 )
    {
      goto LABEL_44;
    }
    v22 = v3[4];
    if ( v22 != 0 && v22 != -8 && v22 != -16 )
      sub_1649B30(v3 + 2);
LABEL_44:
    v3 += 5;
  }
  while ( v3 != v5 );
  v23 = *(unsigned int *)(a1 + 48);
  if ( !v2 )
  {
    if ( (_DWORD)v23 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_DWORD *)(a1 + 48) = 0;
      goto LABEL_18;
    }
LABEL_17:
    *(_QWORD *)(a1 + 40) = 0;
    goto LABEL_18;
  }
  v24 = 64;
  v25 = v2 - 1;
  if ( v25 )
  {
    _BitScanReverse(&v26, v25);
    v24 = 1 << (33 - (v26 ^ 0x1F));
    if ( v24 < 64 )
      v24 = 64;
  }
  v27 = *(_QWORD **)(a1 + 32);
  if ( (_DWORD)v23 == v24 )
  {
    *(_QWORD *)(a1 + 40) = 0;
    v33 = &v27[5 * v23];
    do
    {
      if ( v27 )
      {
        *v27 = -8;
        v27[1] = -8;
      }
      v27 += 5;
    }
    while ( v33 != v27 );
  }
  else
  {
    j___libc_free_0(v27);
    v28 = ((((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 16;
    v29 = (v28
         | (((((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
             | (4 * v24 / 3u + 1)
             | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
           | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 8)
         | (((((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
           | (4 * v24 / 3u + 1)
           | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 4)
         | (((4 * v24 / 3u + 1) | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1)) >> 2)
         | (4 * v24 / 3u + 1)
         | ((unsigned __int64)(4 * v24 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 48) = v29;
    v30 = (_QWORD *)sub_22077B0(40 * v29);
    v31 = *(unsigned int *)(a1 + 48);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = v30;
    for ( i = &v30[5 * v31]; i != v30; v30 += 5 )
    {
      if ( v30 )
      {
        *v30 = -8;
        v30[1] = -8;
      }
    }
  }
LABEL_18:
  sub_1940B30(a1 + 56);
  sub_1940B30(a1 + 88);
  v8 = *(_DWORD *)(a1 + 240);
  ++*(_QWORD *)(a1 + 224);
  if ( v8 )
  {
    v12 = 4 * v8;
    v10 = *(unsigned int *)(a1 + 248);
    if ( (unsigned int)(4 * v8) < 0x40 )
      v12 = 64;
    if ( v12 >= (unsigned int)v10 )
    {
LABEL_21:
      result = *(_QWORD *)(a1 + 232);
      for ( j = result + 8 * v10; j != result; result += 8 )
        *(_QWORD *)result = -8;
      *(_QWORD *)(a1 + 240) = 0;
      return result;
    }
    v13 = *(_QWORD **)(a1 + 232);
    v14 = v8 - 1;
    if ( v14 )
    {
      _BitScanReverse(&v14, v14);
      v15 = 1 << (33 - (v14 ^ 0x1F));
      if ( v15 < 64 )
        v15 = 64;
      if ( (_DWORD)v10 == v15 )
      {
        *(_QWORD *)(a1 + 240) = 0;
        result = (__int64)&v13[v10];
        do
        {
          if ( v13 )
            *v13 = -8;
          ++v13;
        }
        while ( (_QWORD *)result != v13 );
        return result;
      }
      v16 = (((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
          | (4 * v15 / 3u + 1)
          | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)
          | (((((4 * v15 / 3u + 1) | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 2)
            | (4 * v15 / 3u + 1)
            | ((unsigned __int64)(4 * v15 / 3u + 1) >> 1)) >> 4);
      v17 = (v16 >> 8) | v16;
      v18 = (v17 | (v17 >> 16)) + 1;
      v19 = 8 * ((v17 | (v17 >> 16)) + 1);
    }
    else
    {
      v19 = 1024;
      v18 = 128;
    }
    j___libc_free_0(v13);
    *(_DWORD *)(a1 + 248) = v18;
    result = sub_22077B0(v19);
    v20 = *(unsigned int *)(a1 + 248);
    *(_QWORD *)(a1 + 240) = 0;
    *(_QWORD *)(a1 + 232) = result;
    for ( k = result + 8 * v20; k != result; result += 8 )
    {
      if ( result )
        *(_QWORD *)result = -8;
    }
    return result;
  }
  result = *(unsigned int *)(a1 + 244);
  if ( (_DWORD)result )
  {
    v10 = *(unsigned int *)(a1 + 248);
    if ( (unsigned int)v10 <= 0x40 )
      goto LABEL_21;
    result = j___libc_free_0(*(_QWORD *)(a1 + 232));
    *(_QWORD *)(a1 + 232) = 0;
    *(_QWORD *)(a1 + 240) = 0;
    *(_DWORD *)(a1 + 248) = 0;
  }
  return result;
}
