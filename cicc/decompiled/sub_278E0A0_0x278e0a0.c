// Function: sub_278E0A0
// Address: 0x278e0a0
//
void __fastcall sub_278E0A0(__int64 a1)
{
  int v2; // r12d
  unsigned int v3; // eax
  _QWORD *v4; // rbx
  __int64 v5; // rdx
  __int64 v6; // r13
  _QWORD *v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // r15
  __int64 v11; // rax
  __int64 v12; // rbx
  unsigned int v13; // r12d
  unsigned int v14; // eax
  unsigned __int64 *v15; // r12
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // rdx
  _QWORD *i; // rdx
  __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  __int64 v23; // rax
  bool v24; // zf
  _QWORD v25[2]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v26; // [rsp+10h] [rbp-60h]
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  __int64 v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 && !*(_DWORD *)(a1 + 20) )
    return;
  v3 = 4 * v2;
  v4 = *(_QWORD **)(a1 + 8);
  v5 = *(unsigned int *)(a1 + 24);
  v6 = 4 * v5;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v3 = 64;
  if ( (unsigned int)v5 > v3 )
  {
    v25[0] = 0;
    v10 = &v4[v6];
    v25[1] = 0;
    v26 = -4096;
    v27 = 0;
    v28 = 0;
    v29 = -8192;
    do
    {
      v11 = v4[2];
      if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
        sub_BD60C0(v4);
      v4 += 4;
    }
    while ( v10 != v4 );
    if ( v26 != 0 && v26 != -4096 )
      sub_BD60C0(v25);
    if ( v2 )
    {
      v12 = 64;
      v13 = v2 - 1;
      if ( v13 )
      {
        _BitScanReverse(&v14, v13);
        v12 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
        if ( (int)v12 < 64 )
          v12 = 64;
      }
      v15 = *(unsigned __int64 **)(a1 + 8);
      if ( *(_DWORD *)(a1 + 24) == (_DWORD)v12 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        v22 = &v15[4 * v12];
        v27 = 0;
        v28 = 0;
        v29 = -4096;
        if ( v22 != v15 )
        {
          do
          {
            if ( v15 )
            {
              *v15 = 0;
              v15[1] = 0;
              v23 = v29;
              v24 = v29 == -4096;
              v15[2] = v29;
              if ( v23 != 0 && !v24 && v23 != -8192 )
                sub_BD6050(v15, v27 & 0xFFFFFFFFFFFFFFF8LL);
            }
            v15 += 4;
          }
          while ( v22 != v15 );
          if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
            goto LABEL_18;
        }
      }
      else
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), v6 * 8, 8);
        v16 = ((((((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v12 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v12 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v12 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v12 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 16;
        v17 = (v16
             | (((((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v12 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v12 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v12 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v12 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 24) = v17;
        v18 = (_QWORD *)sub_C7D670(32 * v17, 8);
        v19 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 8) = v18;
        for ( i = &v18[4 * v19]; i != v18; v18 += 4 )
        {
          if ( v18 )
          {
            *v18 = 0;
            v18[1] = 0;
            v18[2] = -4096;
          }
        }
      }
      return;
    }
    v21 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) )
    {
      sub_C7D6A0(v21, v6 * 8, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return;
    }
LABEL_37:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v7 = &v4[v6];
  v27 = 0;
  v8 = -4096;
  v28 = 0;
  v29 = -4096;
  if ( v7 == v4 )
    goto LABEL_37;
  do
  {
    v9 = v4[2];
    if ( v9 != v8 )
    {
      if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        sub_BD60C0(v4);
      v4[2] = v8;
      if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
        sub_BD73F0((__int64)v4);
      v8 = v29;
    }
    v4 += 4;
  }
  while ( v4 != v7 );
  *(_QWORD *)(a1 + 16) = 0;
  if ( v8 != -8192 && v8 != -4096 && v8 )
LABEL_18:
    sub_BD60C0(&v27);
}
