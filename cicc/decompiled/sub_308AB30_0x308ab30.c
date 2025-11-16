// Function: sub_308AB30
// Address: 0x308ab30
//
__int64 __fastcall sub_308AB30(__int64 a1)
{
  int v2; // ecx
  __int64 v3; // rax
  _QWORD *v4; // rdi
  _QWORD *i; // rax
  int v6; // ecx
  __int64 result; // rax
  __int64 v8; // rax
  _QWORD *v9; // rdi
  _QWORD *v10; // r13
  _QWORD *v11; // rax
  _QWORD *v12; // rbx
  unsigned __int64 v13; // r14
  unsigned int v14; // edx
  unsigned int v15; // ecx
  unsigned int v16; // edx
  char v17; // cl
  _QWORD *v18; // rdx
  unsigned int v19; // ebx
  _QWORD *v20; // r13
  _QWORD *v21; // rax
  _QWORD *v22; // rbx
  unsigned __int64 v23; // r14
  unsigned int v24; // edx
  unsigned int v25; // ecx
  unsigned int v26; // edx
  char v27; // cl
  _QWORD *v28; // rdx
  int v29; // ebx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *j; // rdx
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  __int64 k; // rdx
  _QWORD *v39; // rax

  v2 = *(_DWORD *)(a1 + 248);
  if ( !v2 )
  {
    ++*(_QWORD *)(a1 + 232);
LABEL_3:
    if ( !*(_DWORD *)(a1 + 252) )
      goto LABEL_9;
    v3 = *(unsigned int *)(a1 + 256);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_C7D6A0(*(_QWORD *)(a1 + 240), 16 * v3, 8);
      *(_QWORD *)(a1 + 240) = 0;
      *(_QWORD *)(a1 + 248) = 0;
      *(_DWORD *)(a1 + 256) = 0;
      goto LABEL_9;
    }
    v4 = *(_QWORD **)(a1 + 240);
    goto LABEL_6;
  }
  v4 = *(_QWORD **)(a1 + 240);
  v20 = &v4[2 * *(unsigned int *)(a1 + 256)];
  if ( v4 == v20 )
    goto LABEL_52;
  v21 = v4;
  while ( 1 )
  {
    v22 = v21;
    if ( *v21 != -4096 && *v21 != -8192 )
      break;
    v21 += 2;
    if ( v20 == v21 )
      goto LABEL_52;
  }
  if ( v21 == v20 )
  {
LABEL_52:
    ++*(_QWORD *)(a1 + 232);
  }
  else
  {
    do
    {
      sub_308A970(v22[1]);
      v23 = v22[1];
      if ( v23 )
      {
        sub_C7D6A0(*(_QWORD *)(v23 + 8), 8LL * *(unsigned int *)(v23 + 24), 4);
        j_j___libc_free_0(v23);
      }
      v22 += 2;
      if ( v22 == v20 )
        break;
      while ( *v22 == -8192 || *v22 == -4096 )
      {
        v22 += 2;
        if ( v20 == v22 )
          goto LABEL_60;
      }
    }
    while ( v20 != v22 );
LABEL_60:
    v2 = *(_DWORD *)(a1 + 248);
    ++*(_QWORD *)(a1 + 232);
    if ( !v2 )
      goto LABEL_3;
    v4 = *(_QWORD **)(a1 + 240);
  }
  v24 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 256);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v24 = 64;
  if ( v24 >= (unsigned int)v3 )
  {
LABEL_6:
    for ( i = &v4[2 * v3]; i != v4; v4 += 2 )
      *v4 = -4096;
    *(_QWORD *)(a1 + 248) = 0;
    goto LABEL_9;
  }
  v25 = v2 - 1;
  if ( !v25 )
  {
    v29 = 64;
LABEL_69:
    sub_C7D6A0((__int64)v4, 16 * v3, 8);
    v30 = ((((((((4 * v29 / 3u + 1) | ((unsigned __int64)(4 * v29 / 3u + 1) >> 1)) >> 2)
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
    v31 = (v30
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
    *(_DWORD *)(a1 + 256) = v31;
    v32 = (_QWORD *)sub_C7D670(16 * v31, 8);
    v33 = *(unsigned int *)(a1 + 256);
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 240) = v32;
    for ( j = &v32[2 * v33]; j != v32; v32 += 2 )
    {
      if ( v32 )
        *v32 = -4096;
    }
    goto LABEL_9;
  }
  _BitScanReverse(&v26, v25);
  v27 = 33 - (v26 ^ 0x1F);
  v28 = v4;
  v29 = 1 << v27;
  if ( 1 << v27 < 64 )
    v29 = 64;
  if ( (_DWORD)v3 != v29 )
    goto LABEL_69;
  *(_QWORD *)(a1 + 248) = 0;
  v39 = &v4[2 * v3];
  do
  {
    if ( v28 )
      *v28 = -4096;
    v28 += 2;
  }
  while ( v39 != v28 );
LABEL_9:
  v6 = *(_DWORD *)(a1 + 280);
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 264);
LABEL_11:
    result = *(unsigned int *)(a1 + 284);
    if ( !(_DWORD)result )
      return result;
    v8 = *(unsigned int *)(a1 + 288);
    if ( (unsigned int)v8 > 0x40 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 272), 16 * v8, 8);
      *(_QWORD *)(a1 + 272) = 0;
      *(_QWORD *)(a1 + 280) = 0;
      *(_DWORD *)(a1 + 288) = 0;
      return result;
    }
    v9 = *(_QWORD **)(a1 + 272);
    goto LABEL_14;
  }
  v9 = *(_QWORD **)(a1 + 272);
  v10 = &v9[2 * *(unsigned int *)(a1 + 288)];
  if ( v9 == v10 )
    goto LABEL_23;
  v11 = *(_QWORD **)(a1 + 272);
  while ( 1 )
  {
    v12 = v11;
    if ( *v11 != -8192 && *v11 != -4096 )
      break;
    v11 += 2;
    if ( v10 == v11 )
      goto LABEL_23;
  }
  if ( v11 == v10 )
  {
LABEL_23:
    ++*(_QWORD *)(a1 + 264);
  }
  else
  {
    do
    {
      sub_308A970(v12[1]);
      v13 = v12[1];
      if ( v13 )
      {
        sub_C7D6A0(*(_QWORD *)(v13 + 8), 8LL * *(unsigned int *)(v13 + 24), 4);
        j_j___libc_free_0(v13);
      }
      v12 += 2;
      if ( v12 == v10 )
        break;
      while ( *v12 == -8192 || *v12 == -4096 )
      {
        v12 += 2;
        if ( v10 == v12 )
          goto LABEL_31;
      }
    }
    while ( v12 != v10 );
LABEL_31:
    v6 = *(_DWORD *)(a1 + 280);
    ++*(_QWORD *)(a1 + 264);
    if ( !v6 )
      goto LABEL_11;
    v9 = *(_QWORD **)(a1 + 272);
  }
  v14 = 4 * v6;
  v8 = *(unsigned int *)(a1 + 288);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v14 = 64;
  if ( v14 >= (unsigned int)v8 )
  {
LABEL_14:
    for ( result = (__int64)&v9[2 * v8]; (_QWORD *)result != v9; v9 += 2 )
      *v9 = -4096;
    *(_QWORD *)(a1 + 280) = 0;
    return result;
  }
  v15 = v6 - 1;
  if ( v15 )
  {
    _BitScanReverse(&v16, v15);
    v17 = 33 - (v16 ^ 0x1F);
    v18 = v9;
    v19 = 1 << v17;
    if ( 1 << v17 < 64 )
      v19 = 64;
    if ( v19 == (_DWORD)v8 )
    {
      *(_QWORD *)(a1 + 280) = 0;
      result = (__int64)&v9[2 * v19];
      do
      {
        if ( v18 )
          *v18 = -4096;
        v18 += 2;
      }
      while ( (_QWORD *)result != v18 );
      return result;
    }
  }
  else
  {
    v19 = 64;
  }
  sub_C7D6A0((__int64)v9, 16 * v8, 8);
  v35 = ((((((((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
           | (4 * v19 / 3 + 1)
           | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 4)
         | (((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
         | (4 * v19 / 3 + 1)
         | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 8)
       | (((((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
         | (4 * v19 / 3 + 1)
         | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 4)
       | (((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
       | (4 * v19 / 3 + 1)
       | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 16;
  v36 = (v35
       | (((((((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
           | (4 * v19 / 3 + 1)
           | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 4)
         | (((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
         | (4 * v19 / 3 + 1)
         | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 8)
       | (((((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
         | (4 * v19 / 3 + 1)
         | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 4)
       | (((4 * v19 / 3 + 1) | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1)) >> 2)
       | (4 * v19 / 3 + 1)
       | ((unsigned __int64)(4 * v19 / 3 + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 288) = v36;
  result = sub_C7D670(16 * v36, 8);
  v37 = *(unsigned int *)(a1 + 288);
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 272) = result;
  for ( k = result + 16 * v37; k != result; result += 16 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
  return result;
}
