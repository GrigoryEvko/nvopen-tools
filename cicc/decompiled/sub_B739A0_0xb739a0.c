// Function: sub_B739A0
// Address: 0xb739a0
//
void __fastcall sub_B739A0(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // r13
  int *v4; // rbx
  int *v5; // r13
  int v6; // r12d
  int *v7; // r8
  __int64 v8; // r12
  bool v9; // cc
  unsigned int v10; // eax
  char v11; // al
  unsigned int v12; // eax
  int v13; // r12d
  unsigned int v14; // ebx
  unsigned int v15; // eax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdi
  __int64 v22; // r13
  __int64 v23; // rbx
  unsigned int v24; // eax
  int *v25; // [rsp+8h] [rbp-58h]
  int *v26; // [rsp+8h] [rbp-58h]
  int *v27; // [rsp+8h] [rbp-58h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 v29; // [rsp+18h] [rbp-48h] BYREF
  unsigned int v30; // [rsp+20h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_B738D0(a1);
      if ( *(_DWORD *)(a1 + 24) )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * (unsigned int)v3, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_48;
    }
LABEL_4:
    v4 = *(int **)(a1 + 8);
    v5 = &v4[8 * v3];
    v30 = 0;
    v29 = -1;
    if ( v4 != v5 )
    {
      v6 = *v4;
      v7 = v4 + 2;
      if ( *v4 == -1 )
        goto LABEL_15;
LABEL_6:
      if ( v6 != -2 || *((_BYTE *)v4 + 4) || v4[4] || *((_QWORD *)v4 + 1) != -2 )
      {
        v8 = *((_QWORD *)v4 + 3);
        if ( v8 )
        {
          if ( *(_DWORD *)(v8 + 32) > 0x40u )
          {
            v21 = *(_QWORD *)(v8 + 24);
            if ( v21 )
            {
              v27 = v7;
              j_j___libc_free_0_0(v21);
              v7 = v27;
            }
          }
          v25 = v7;
          sub_BD7260(v8);
          sub_BD2DD0(v8);
          v7 = v25;
        }
        v9 = (unsigned int)v4[4] <= 0x40;
        *v4 = -1;
        *((_BYTE *)v4 + 4) = 1;
        if ( !v9 )
          goto LABEL_12;
      }
      else
      {
        *v4 = -1;
        *((_BYTE *)v4 + 4) = 1;
      }
      if ( v30 > 0x40 )
      {
LABEL_12:
        sub_C43990(v7, &v29);
        goto LABEL_13;
      }
      *((_QWORD *)v4 + 1) = v29;
      v4[4] = v30;
LABEL_13:
      while ( 1 )
      {
        v4 += 8;
        if ( v5 == v4 )
          break;
        v6 = *v4;
        v7 = v4 + 2;
        if ( *v4 != -1 )
          goto LABEL_6;
LABEL_15:
        if ( *((_BYTE *)v4 + 4) != 1 )
          goto LABEL_6;
        v10 = v4[4];
        if ( v10 != v30 )
          goto LABEL_6;
        if ( v10 <= 0x40 )
        {
          if ( *((_QWORD *)v4 + 1) != v29 )
            goto LABEL_6;
        }
        else
        {
          v26 = v7;
          v11 = sub_C43C50(v7, &v29);
          v7 = v26;
          if ( !v11 )
            goto LABEL_6;
        }
      }
      *(_QWORD *)(a1 + 16) = 0;
LABEL_21:
      if ( v30 > 0x40 && v29 )
        j_j___libc_free_0_0(v29);
      return;
    }
LABEL_48:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v12 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v12 = 64;
  if ( v12 >= (unsigned int)v3 )
    goto LABEL_4;
  v13 = 64;
  sub_B738D0(a1);
  v14 = v2 - 1;
  if ( v14 )
  {
    _BitScanReverse(&v15, v14);
    v13 = 1 << (33 - (v15 ^ 0x1F));
    if ( v13 < 64 )
      v13 = 64;
  }
  if ( *(_DWORD *)(a1 + 24) == v13 )
  {
    v22 = *(_QWORD *)(a1 + 8);
    BYTE4(v28) = 1;
    *(_QWORD *)(a1 + 16) = 0;
    v23 = v22 + 32LL * (unsigned int)v13;
    LODWORD(v28) = -1;
    v30 = 0;
    v29 = -1;
    if ( v22 == v23 )
      return;
    do
    {
      if ( v22 )
      {
        *(_QWORD *)v22 = v28;
        v24 = v30;
        *(_DWORD *)(v22 + 16) = v30;
        if ( v24 <= 0x40 )
          *(_QWORD *)(v22 + 8) = v29;
        else
          sub_C43780(v22 + 8, &v29);
      }
      v22 += 32;
    }
    while ( v23 != v22 );
    goto LABEL_21;
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 8), 32LL * (unsigned int)v3, 8);
  v16 = ((((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
         | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
       | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
       | (4 * v13 / 3u + 1)
       | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 16;
  v17 = (v16
       | (((((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
           | (4 * v13 / 3u + 1)
           | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
         | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 8)
       | (((((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
         | (4 * v13 / 3u + 1)
         | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 4)
       | (((4 * v13 / 3u + 1) | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1)) >> 2)
       | (4 * v13 / 3u + 1)
       | ((unsigned __int64)(4 * v13 / 3u + 1) >> 1))
      + 1;
  *(_DWORD *)(a1 + 24) = v17;
  v18 = sub_C7D670(32 * v17, 8);
  v19 = *(unsigned int *)(a1 + 24);
  LODWORD(v28) = -1;
  *(_QWORD *)(a1 + 8) = v18;
  *(_QWORD *)(a1 + 16) = 0;
  v20 = v18 + 32 * v19;
  for ( BYTE4(v28) = 1; v20 != v18; v18 += 32 )
  {
    if ( v18 )
    {
      *(_DWORD *)(v18 + 16) = 0;
      *(_QWORD *)(v18 + 8) = -1;
      *(_QWORD *)v18 = v28;
    }
  }
}
