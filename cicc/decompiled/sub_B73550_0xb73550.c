// Function: sub_B73550
// Address: 0xb73550
//
void __fastcall sub_B73550(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edx
  __int64 v7; // r8
  __int64 v8; // rdi
  unsigned int v9; // r12d
  char v10; // al
  unsigned int v11; // eax
  __int64 v12; // r15
  unsigned int v13; // ebx
  unsigned int v14; // eax
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 i; // rdx
  __int64 v20; // rbx
  __int64 v21; // r12
  unsigned int v22; // eax
  __int64 v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+8h] [rbp-48h]
  unsigned int v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v27; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_B73490(a1);
      if ( *(_DWORD *)(a1 + 24) )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), 24 * v3, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_43;
    }
LABEL_4:
    v4 = *(_QWORD *)(a1 + 8);
    v27 = 0;
    v26 = -1;
    v5 = v4 + 24 * v3;
    if ( v4 != v5 )
    {
      v6 = 0;
      while ( 1 )
      {
        v9 = *(_DWORD *)(v4 + 8);
        if ( v9 != v6 )
          break;
        if ( v6 <= 0x40 )
        {
          if ( *(_QWORD *)v4 == v26 )
            goto LABEL_16;
          if ( v9 )
          {
LABEL_7:
            v7 = *(_QWORD *)(v4 + 16);
            if ( v7 )
            {
LABEL_8:
              if ( *(_DWORD *)(v7 + 32) > 0x40u )
              {
                v8 = *(_QWORD *)(v7 + 24);
                if ( v8 )
                {
                  v23 = v7;
                  j_j___libc_free_0_0(v8);
                  v7 = v23;
                }
              }
              v24 = v7;
              sub_BD7260(v7);
              sub_BD2DD0(v24);
              v9 = *(_DWORD *)(v4 + 8);
            }
            if ( v9 > 0x40 )
              goto LABEL_15;
LABEL_13:
            v6 = v27;
            goto LABEL_14;
          }
LABEL_40:
          if ( *(_QWORD *)v4 != -2 )
          {
            v7 = *(_QWORD *)(v4 + 16);
            if ( v7 )
              goto LABEL_8;
            goto LABEL_13;
          }
LABEL_14:
          if ( v6 <= 0x40 )
          {
            *(_QWORD *)v4 = v26;
            v6 = v27;
            *(_DWORD *)(v4 + 8) = v27;
            goto LABEL_16;
          }
LABEL_15:
          sub_C43990(v4, &v26);
          v6 = v27;
LABEL_16:
          v4 += 24;
          if ( v5 == v4 )
            goto LABEL_21;
        }
        else
        {
          v25 = v6;
          v10 = sub_C43C50(v4, &v26);
          v6 = v25;
          if ( !v10 )
            break;
          v4 += 24;
          if ( v5 == v4 )
          {
LABEL_21:
            *(_QWORD *)(a1 + 16) = 0;
            if ( v6 > 0x40 )
              goto LABEL_22;
            return;
          }
        }
      }
      if ( v9 )
        goto LABEL_7;
      goto LABEL_40;
    }
LABEL_43:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v11 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v11 = 64;
  if ( v11 >= (unsigned int)v3 )
    goto LABEL_4;
  v12 = 64;
  sub_B73490(a1);
  v13 = v2 - 1;
  if ( v13 )
  {
    _BitScanReverse(&v14, v13);
    v12 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
    if ( (int)v12 < 64 )
      v12 = 64;
  }
  if ( *(_DWORD *)(a1 + 24) == (_DWORD)v12 )
  {
    v20 = *(_QWORD *)(a1 + 8);
    *(_QWORD *)(a1 + 16) = 0;
    v27 = 0;
    v21 = v20 + 24 * v12;
    v26 = -1;
    if ( v20 != v21 )
    {
      do
      {
        if ( v20 )
        {
          v22 = v27;
          *(_DWORD *)(v20 + 8) = v27;
          if ( v22 <= 0x40 )
            *(_QWORD *)v20 = v26;
          else
            sub_C43780(v20, &v26);
        }
        v20 += 24;
      }
      while ( v21 != v20 );
      if ( v27 > 0x40 )
      {
LABEL_22:
        if ( v26 )
          j_j___libc_free_0_0(v26);
      }
    }
  }
  else
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 8), 24 * v3, 8);
    v15 = ((((((((4 * (int)v12 / 3u + 1) | ((unsigned __int64)(4 * (int)v12 / 3u + 1) >> 1)) >> 2)
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
    v16 = (v15
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
    *(_DWORD *)(a1 + 24) = v16;
    v17 = sub_C7D670(24 * v16, 8);
    v18 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v17;
    for ( i = v17 + 24 * v18; i != v17; v17 += 24 )
    {
      if ( v17 )
      {
        *(_DWORD *)(v17 + 8) = 0;
        *(_QWORD *)v17 = -1;
      }
    }
  }
}
