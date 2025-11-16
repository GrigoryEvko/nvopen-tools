// Function: sub_FDCC00
// Address: 0xfdcc00
//
void __fastcall sub_FDCC00(__int64 a1)
{
  int v2; // r15d
  __int64 v3; // rax
  _QWORD *v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // edx
  unsigned int v7; // eax
  __int64 v8; // r12
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 i; // rax
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // rbx
  unsigned int v19; // r15d
  unsigned int v20; // eax
  unsigned __int64 *v21; // r14
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rax
  _QWORD *v24; // rax
  __int64 v25; // rdx
  _QWORD *j; // rdx
  __int64 v27; // rdi
  unsigned __int64 *v28; // rbx
  __int64 v29; // rax
  bool v30; // zf
  _QWORD v31[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
  v31[0] = 0;
  v31[1] = 0;
  v5 = 9 * v3;
  v6 = v3;
  v7 = 4 * v2;
  v32 = -4096;
  v8 = 8 * v5;
  v33 = 0;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v7 = 64;
  v34 = 0;
  v9 = &v4[(unsigned __int64)v8 / 8];
  v35 = -8192;
  if ( v6 <= v7 )
  {
    v10 = -8192;
    v11 = -4096;
    if ( v4 == v9 )
      goto LABEL_57;
    while ( 1 )
    {
      v13 = v4[2];
      if ( v13 != v11 )
        break;
LABEL_12:
      v4 += 9;
      if ( v4 == v9 )
      {
        if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
          sub_BD60C0(&v33);
        v14 = v32;
        *(_QWORD *)(a1 + 16) = 0;
        if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
          sub_BD60C0(v31);
        return;
      }
      v11 = v32;
    }
    if ( v13 != v10 )
    {
      v4[4] = &unk_49DB368;
      v12 = v4[7];
      if ( v12 == -4096 || v12 == 0 || v12 == -8192 )
      {
        v10 = v13;
      }
      else
      {
        sub_BD60C0(v4 + 5);
        v11 = v32;
        v10 = v4[2];
        if ( v32 == v10 )
        {
LABEL_11:
          v10 = v35;
          goto LABEL_12;
        }
      }
    }
    if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
      sub_BD60C0(v4);
    v4[2] = v11;
    if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
      sub_BD73F0((__int64)v4);
    goto LABEL_11;
  }
  for ( i = -4096; ; i = v32 )
  {
    v16 = v4[2];
    if ( v16 != i )
    {
      i = v35;
      if ( v16 != v35 )
      {
        v4[4] = &unk_49DB368;
        v17 = v4[7];
        if ( v17 != -4096 && v17 != 0 && v17 != -8192 )
        {
          sub_BD60C0(v4 + 5);
          v16 = v4[2];
        }
        i = v16;
      }
    }
    if ( i != -4096 && i != 0 && i != -8192 )
      sub_BD60C0(v4);
    v4 += 9;
    if ( v4 == v9 )
      break;
  }
  if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
    sub_BD60C0(&v33);
  if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
    sub_BD60C0(v31);
  if ( !v2 )
  {
    v27 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(v27, v8, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
LABEL_57:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v18 = 64;
  v19 = v2 - 1;
  if ( v19 )
  {
    _BitScanReverse(&v20, v19);
    v18 = (unsigned int)(1 << (33 - (v20 ^ 0x1F)));
    if ( (int)v18 < 64 )
      v18 = 64;
  }
  v21 = *(unsigned __int64 **)(a1 + 8);
  if ( *(_DWORD *)(a1 + 24) == (_DWORD)v18 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = &v21[9 * v18];
    v33 = 0;
    v34 = 0;
    v35 = -4096;
    if ( v28 != v21 )
    {
      do
      {
        if ( v21 )
        {
          *v21 = 0;
          v21[1] = 0;
          v29 = v35;
          v30 = v35 == 0;
          v21[2] = v35;
          if ( v29 != -4096 && !v30 && v29 != -8192 )
            sub_BD6050(v21, v33 & 0xFFFFFFFFFFFFFFF8LL);
        }
        v21 += 9;
      }
      while ( v28 != v21 );
      if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
        sub_BD60C0(&v33);
    }
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), v8, 8);
    v22 = ((((((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v18 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v18 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 16;
    v23 = (v22
         | (((((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v18 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v18 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v18 / 3u + 1) | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v18 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v18 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 24) = v23;
    v24 = (_QWORD *)sub_C7D670(72 * v23, 8);
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v24;
    for ( j = &v24[9 * v25]; j != v24; v24 += 9 )
    {
      if ( v24 )
      {
        *v24 = 0;
        v24[1] = 0;
        v24[2] = -4096;
      }
    }
  }
}
