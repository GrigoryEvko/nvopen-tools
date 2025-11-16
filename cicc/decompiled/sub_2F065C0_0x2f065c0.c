// Function: sub_2F065C0
// Address: 0x2f065c0
//
__int64 __fastcall sub_2F065C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rax
  __int64 v7; // rcx
  unsigned int v8; // r14d
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  unsigned __int64 *v13; // rdx
  __int64 i; // rsi
  __int64 v15; // rax
  __int64 j; // rsi
  __int64 v17; // r8
  __int64 v18; // rcx
  __int64 v19; // r9
  __int64 k; // r15
  __int64 v21; // rax
  unsigned __int64 v22; // r10
  _QWORD *v23; // rax
  _QWORD *v24; // rdi
  __int64 v25; // r15
  __int64 m; // rcx
  __int64 v27; // rax
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rax
  _QWORD *v30; // r9
  __int64 v31; // r12
  __int64 n; // r15
  __int64 v33; // [rsp+0h] [rbp-60h]
  __int64 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v37; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v38; // [rsp+18h] [rbp-48h]
  unsigned __int64 *v39; // [rsp+18h] [rbp-48h]
  unsigned __int64 v40; // [rsp+20h] [rbp-40h] BYREF
  __int64 v41; // [rsp+28h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 120);
  v7 = v6 + 16LL * *(unsigned int *)(a2 + 128);
  if ( v6 != v7 )
  {
    while ( (((unsigned __int8)*(_QWORD *)v6 ^ 6) & 6) != 0 || *(_DWORD *)(v6 + 8) != 5 )
    {
      v6 += 16;
      if ( v7 == v6 )
        goto LABEL_8;
    }
    return 0;
  }
LABEL_8:
  v10 = *(_QWORD *)(a3 + 40);
  v11 = v10 + 16LL * *(unsigned int *)(a3 + 48);
  if ( v10 != v11 )
  {
    while ( (((unsigned __int8)*(_QWORD *)v10 ^ 6) & 6) != 0 || *(_DWORD *)(v10 + 8) != 5 )
    {
      v10 += 16;
      if ( v11 == v10 )
        goto LABEL_14;
    }
    return 0;
  }
LABEL_14:
  v41 = 5;
  v40 = a2 | 6;
  v8 = sub_2F92BA0(a1, a3, &v40);
  if ( !(_BYTE)v8 )
    return 0;
  v12 = *(_QWORD *)(a2 + 120);
  v13 = &v40;
  for ( i = v12 + 16LL * *(unsigned int *)(a2 + 128); i != v12; v12 += 16 )
  {
    if ( a3 == (*(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL) )
      *(_DWORD *)(v12 + 12) = 0;
  }
  v15 = *(_QWORD *)(a3 + 40);
  for ( j = v15 + 16LL * *(unsigned int *)(a3 + 48); j != v15; v15 += 16 )
  {
    if ( a2 == (*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL) )
      *(_DWORD *)(v15 + 12) = 0;
  }
  v17 = a1 + 328;
  if ( a3 != a1 + 328 )
  {
    v18 = *(_QWORD *)(a2 + 120);
    v19 = 16LL * *(unsigned int *)(a2 + 128);
    for ( k = v18 + v19; k != v18; v18 += 16 )
    {
      v21 = (*(__int64 *)v18 >> 1) & 3;
      if ( v21 == 3 )
      {
        if ( *(_DWORD *)(v18 + 8) <= 3u )
        {
LABEL_29:
          v22 = *(_QWORD *)v18 & 0xFFFFFFFFFFFFFFF8LL;
          if ( a3 != v22 && v17 != v22 && v21 != 2 )
          {
            v23 = *(_QWORD **)(v22 + 40);
            v24 = &v23[2 * *(unsigned int *)(v22 + 48)];
            if ( v23 == v24 )
            {
LABEL_58:
              v33 = v18;
              v35 = v17;
              v37 = v13;
              v40 = a3 & 0xFFFFFFFFFFFFFFF9LL | 6;
              v41 = 3;
              sub_2F92BA0(a1, v22, v13);
              v13 = v37;
              v17 = v35;
              v18 = v33;
            }
            else
            {
              while ( a3 != (*v23 & 0xFFFFFFFFFFFFFFF8LL) )
              {
                v23 += 2;
                if ( v24 == v23 )
                  goto LABEL_58;
              }
            }
          }
        }
      }
      else if ( v21 != 1 )
      {
        goto LABEL_29;
      }
    }
  }
  if ( a2 != a1 + 72 )
  {
    v25 = *(_QWORD *)(a3 + 40);
    for ( m = v25 + 16LL * *(unsigned int *)(a3 + 48); m != v25; v25 += 16 )
    {
      v27 = (*(__int64 *)v25 >> 1) & 3;
      if ( v27 == 3 )
      {
        if ( *(_DWORD *)(v25 + 8) <= 3u )
        {
LABEL_40:
          v28 = *(_QWORD *)v25 & 0xFFFFFFFFFFFFFFF8LL;
          if ( v27 != 2 && a2 != v28 )
          {
            v29 = *(_QWORD **)(a2 + 120);
            v30 = &v29[2 * *(unsigned int *)(a2 + 128)];
            if ( v29 == v30 )
            {
LABEL_57:
              v34 = m;
              v40 = v28 | 6;
              v36 = v17;
              v39 = v13;
              v41 = 3;
              sub_2F92BA0(a1, a2, v13);
              v13 = v39;
              v17 = v36;
              m = v34;
            }
            else
            {
              while ( v28 != (*v29 & 0xFFFFFFFFFFFFFFF8LL) )
              {
                v29 += 2;
                if ( v30 == v29 )
                  goto LABEL_57;
              }
            }
          }
        }
      }
      else if ( v27 != 1 )
      {
        goto LABEL_40;
      }
    }
    if ( a3 == v17 )
    {
      v31 = *(_QWORD *)(a1 + 48);
      for ( n = *(_QWORD *)(a1 + 56); n != v31; v31 += 256 )
      {
        if ( !*(_DWORD *)(v31 + 128) )
        {
          v38 = v13;
          v41 = 3;
          v40 = v31 | 6;
          sub_2F92BA0(a1, a2, v13);
          v13 = v38;
        }
      }
    }
  }
  return v8;
}
