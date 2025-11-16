// Function: sub_2F92740
// Address: 0x2f92740
//
unsigned __int64 __fastcall sub_2F92740(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rax
  void *v8; // rdi
  int v9; // edx
  int v10; // ecx
  unsigned __int64 v11; // rcx
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned __int64 v14; // r15
  unsigned __int64 result; // rax
  __int16 v16; // ax
  unsigned __int64 *v17; // rax
  unsigned __int64 *v18; // rdx
  unsigned __int64 i; // rbx
  __int64 v20; // rdx
  __int64 v21; // r10
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rsi
  unsigned int v25; // eax
  unsigned int v26; // ecx
  __int64 v27; // rsi
  __int64 v28; // r10
  __int64 v29; // rax
  unsigned __int64 j; // rbx
  __int16 v31; // ax
  __int64 v32; // r12
  __int64 v33; // [rsp+0h] [rbp-50h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 *v35; // [rsp+10h] [rbp-40h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v6 = *(_QWORD *)(a1 + 24);
  v35 = (__int64 *)(a1 + 3376);
  *(_QWORD *)(a1 + 3376) = v6;
  v7 = *(unsigned int *)(a1 + 3392);
  v8 = *(void **)(a1 + 3384);
  if ( 8 * v7 )
  {
    memset(v8, 0, 8 * v7);
    v7 = *(unsigned int *)(a1 + 3392);
  }
  v9 = *(_DWORD *)(v6 + 44);
  v10 = *(_DWORD *)(a1 + 3448) & 0x3F;
  if ( v10 )
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 3384) + 8 * v7 - 8) &= ~(-1LL << v10);
    v7 = *(unsigned int *)(a1 + 3392);
  }
  *(_DWORD *)(a1 + 3448) = v9;
  v11 = (unsigned int)(v9 + 63) >> 6;
  v12 = v11;
  if ( v11 != v7 )
  {
    if ( v11 >= v7 )
    {
      v32 = v11 - v7;
      if ( v11 > *(unsigned int *)(a1 + 3396) )
      {
        sub_C8D5F0(a1 + 3384, (const void *)(a1 + 3400), v11, 8u, a5, v11);
        v7 = *(unsigned int *)(a1 + 3392);
      }
      if ( 8 * v32 )
      {
        memset((void *)(*(_QWORD *)(a1 + 3384) + 8 * v7), 0, 8 * v32);
        LODWORD(v7) = *(_DWORD *)(a1 + 3392);
      }
      v9 = *(_DWORD *)(a1 + 3448);
      *(_DWORD *)(a1 + 3392) = v32 + v7;
    }
    else
    {
      *(_DWORD *)(a1 + 3392) = v11;
    }
  }
  v13 = v9 & 0x3F;
  if ( (_DWORD)v13 )
  {
    v11 = (unsigned int)v13;
    *(_QWORD *)(*(_QWORD *)(a1 + 3384) + 8LL * *(unsigned int *)(a1 + 3392) - 8) &= ~(-1LL << v13);
  }
  sub_2E225E0(v35, a2, v13, v11, a5, v12);
  v36 = a2 + 48;
  v14 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
    BUG();
  result = *(_QWORD *)v14;
  if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      result &= 0xFFFFFFFFFFFFFFF8LL;
      v14 = result;
      if ( (*(_BYTE *)(result + 44) & 4) == 0 )
        break;
      result = *(_QWORD *)result;
    }
  }
  while ( v36 != v14 )
  {
    v16 = *(_WORD *)(v14 + 68);
    if ( (unsigned __int16)(v16 - 14) <= 4u || v16 == 24 )
      goto LABEL_15;
    for ( i = v14; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v20 = *(_QWORD *)(v14 + 24) + 48LL;
    while ( 1 )
    {
      v21 = *(_QWORD *)(i + 32);
      v22 = v21 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
      if ( v21 != v22 )
        break;
      i = *(_QWORD *)(i + 8);
      if ( v20 == i )
        break;
      if ( (*(_BYTE *)(i + 44) & 4) == 0 )
      {
        i = *(_QWORD *)(v14 + 24) + 48LL;
        break;
      }
    }
    if ( v22 != v21 )
    {
      while ( *(_BYTE *)v21 )
      {
        if ( *(_BYTE *)v21 != 12 )
          goto LABEL_33;
        v33 = v20;
        v34 = v21;
        sub_2E21E10(v35, *(_QWORD *)(v21 + 24));
        v20 = v33;
        v29 = v22;
        v28 = v34 + 40;
        if ( v34 + 40 == v22 )
        {
          while ( 1 )
          {
LABEL_37:
            i = *(_QWORD *)(i + 8);
            if ( v20 == i )
            {
              v21 = v22;
              v22 = v29;
              goto LABEL_39;
            }
            if ( (*(_BYTE *)(i + 44) & 4) == 0 )
              break;
            v22 = *(_QWORD *)(i + 32);
            v29 = v22 + 40LL * (*(_DWORD *)(i + 40) & 0xFFFFFF);
            if ( v22 != v29 )
              goto LABEL_53;
          }
          v21 = v22;
          i = v20;
          v22 = v29;
LABEL_39:
          if ( v22 == v21 )
            goto LABEL_40;
        }
        else
        {
LABEL_52:
          v22 = v28;
LABEL_53:
          v21 = v22;
          v22 = v29;
        }
      }
      if ( (*(_BYTE *)(v21 + 3) & 0x10) != 0 )
      {
        v23 = *(unsigned int *)(v21 + 8);
        if ( (_DWORD)v23 )
        {
          v24 = *(_QWORD *)(a1 + 3376);
          v25 = *(_DWORD *)(*(_QWORD *)(v24 + 8) + 24 * v23 + 16);
          v26 = v25 & 0xFFF;
          v27 = *(_QWORD *)(v24 + 56) + 2LL * (v25 >> 12);
          do
          {
            if ( !v27 )
              break;
            v27 += 2;
            *(_QWORD *)(*(_QWORD *)(a1 + 3384) + 8LL * (v26 >> 6)) &= ~(1LL << v26);
            v26 += *(__int16 *)(v27 - 2);
          }
          while ( *(_WORD *)(v27 - 2) );
        }
      }
LABEL_33:
      v28 = v21 + 40;
      v29 = v22;
      if ( v28 == v22 )
        goto LABEL_37;
      goto LABEL_52;
    }
LABEL_40:
    if ( (*(_BYTE *)(v14 + 44) & 0xC) != 0 )
    {
      if ( *(_WORD *)(v14 + 68) == 21 )
        sub_2F91510(*(_QWORD *)(a1 + 40), v35, v14);
      for ( j = *(_QWORD *)(v14 + 8); (*(_BYTE *)(j + 44) & 8) != 0; j = *(_QWORD *)(j + 8) )
        ;
      do
      {
        v31 = *(_WORD *)(j + 68);
        if ( (unsigned __int16)(v31 - 14) > 4u && v31 != 24 )
          sub_2F91510(*(_QWORD *)(a1 + 40), v35, j);
        j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL;
      }
      while ( j != v14 );
    }
    else
    {
      sub_2F91510(*(_QWORD *)(a1 + 40), v35, v14);
    }
LABEL_15:
    v17 = (unsigned __int64 *)(*(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL);
    v18 = v17;
    if ( !v17 )
      BUG();
    v14 = *(_QWORD *)v14 & 0xFFFFFFFFFFFFFFF8LL;
    result = *v17;
    if ( (result & 4) == 0 && (*((_BYTE *)v18 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        result &= 0xFFFFFFFFFFFFFFF8LL;
        v14 = result;
        if ( (*(_BYTE *)(result + 44) & 4) == 0 )
          break;
        result = *(_QWORD *)result;
      }
    }
  }
  return result;
}
