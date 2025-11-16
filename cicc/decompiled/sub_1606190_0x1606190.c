// Function: sub_1606190
// Address: 0x1606190
//
void __fastcall sub_1606190(__int64 a1)
{
  int v2; // ebx
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // edx
  __int64 v7; // r8
  __int64 v8; // rdi
  unsigned int v9; // r12d
  char v10; // al
  unsigned int v11; // edx
  __int64 v12; // r14
  unsigned int v13; // ebx
  unsigned int v14; // eax
  __int64 v15; // r12
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 i; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rbx
  unsigned int v26; // eax
  __int64 v27; // [rsp+8h] [rbp-48h]
  __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned int v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v31; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v3 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v3 > 0x40 )
    {
      sub_16060D0(a1);
      if ( *(_DWORD *)(a1 + 24) )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      goto LABEL_46;
    }
LABEL_4:
    v4 = *(_QWORD *)(a1 + 8);
    v31 = 0;
    v30 = 0;
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
          if ( *(_QWORD *)v4 == v30 )
            goto LABEL_16;
          if ( v9 )
          {
LABEL_7:
            v7 = *(_QWORD *)(v4 + 16);
            if ( !v7 )
              goto LABEL_12;
            goto LABEL_8;
          }
LABEL_42:
          if ( *(_QWORD *)v4 == 1 )
            goto LABEL_14;
          v7 = *(_QWORD *)(v4 + 16);
          if ( !v7 )
          {
LABEL_13:
            v6 = v31;
LABEL_14:
            if ( v6 <= 0x40 )
            {
              v21 = v30;
              *(_QWORD *)v4 = v30;
              v22 = v31;
              *(_DWORD *)(v4 + 8) = v31;
              v23 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v22;
              if ( (unsigned int)v22 > 0x40 )
              {
                v24 = (unsigned int)((unsigned __int64)(v22 + 63) >> 6) - 1;
                *(_QWORD *)(v21 + 8 * v24) &= v23;
              }
              else
              {
                *(_QWORD *)v4 = v23 & v21;
              }
              v6 = v31;
              goto LABEL_16;
            }
            goto LABEL_15;
          }
LABEL_8:
          if ( *(_DWORD *)(v7 + 32) > 0x40u )
          {
            v8 = *(_QWORD *)(v7 + 24);
            if ( v8 )
            {
              v27 = v7;
              j_j___libc_free_0_0(v8);
              v7 = v27;
            }
          }
          v28 = v7;
          sub_164BE60(v7);
          sub_1648B90(v28);
          v9 = *(_DWORD *)(v4 + 8);
LABEL_12:
          if ( v9 <= 0x40 )
            goto LABEL_13;
LABEL_15:
          sub_16A51C0(v4, &v30);
          v6 = v31;
LABEL_16:
          v4 += 24;
          if ( v5 == v4 )
            goto LABEL_21;
        }
        else
        {
          v29 = v6;
          v10 = sub_16A5220(v4, &v30);
          v6 = v29;
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
      goto LABEL_42;
    }
LABEL_46:
    *(_QWORD *)(a1 + 16) = 0;
    return;
  }
  v11 = 4 * v2;
  v3 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v2) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v3 <= v11 )
    goto LABEL_4;
  v12 = 64;
  sub_16060D0(a1);
  v13 = v2 - 1;
  if ( v13 )
  {
    _BitScanReverse(&v14, v13);
    v12 = (unsigned int)(1 << (33 - (v14 ^ 0x1F)));
    if ( (int)v12 < 64 )
      v12 = 64;
  }
  v15 = *(_QWORD *)(a1 + 8);
  if ( (_DWORD)v12 == *(_DWORD *)(a1 + 24) )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v31 = 0;
    v25 = v15 + 24 * v12;
    v30 = 0;
    do
    {
      if ( v15 )
      {
        v26 = v31;
        *(_DWORD *)(v15 + 8) = v31;
        if ( v26 <= 0x40 )
          *(_QWORD *)v15 = v30;
        else
          sub_16A4FD0(v15, &v30);
      }
      v15 += 24;
    }
    while ( v25 != v15 );
    if ( v31 > 0x40 )
    {
LABEL_22:
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
  }
  else
  {
    j___libc_free_0(*(_QWORD *)(a1 + 8));
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
    v18 = sub_22077B0(24 * v17);
    v19 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 8) = v18;
    for ( i = v18 + 24 * v19; i != v18; v18 += 24 )
    {
      if ( v18 )
      {
        *(_DWORD *)(v18 + 8) = 0;
        *(_QWORD *)v18 = 0;
      }
    }
  }
}
