// Function: sub_F19BD0
// Address: 0xf19bd0
//
unsigned __int64 __fastcall sub_F19BD0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // r14
  __int64 v6; // rbx
  unsigned int v7; // r15d
  __int64 v8; // r12
  __int64 v9; // rcx
  char v10; // di
  __int64 v11; // r8
  int v12; // esi
  int v13; // r11d
  __int64 *v14; // r10
  __int64 *v15; // rdx
  __int64 v16; // r9
  unsigned __int64 v17; // r13
  unsigned int v18; // esi
  unsigned int v19; // eax
  unsigned int v20; // eax
  int v21; // r8d
  unsigned int v22; // r10d
  __int64 v23; // rdx
  __int64 *v24; // r14
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // r12
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdx
  __int64 v32; // r10
  int v33; // r9d
  int v34; // r8d
  __int64 *v35; // rsi
  unsigned int i; // eax
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // r10
  int v40; // r9d
  int v41; // r8d
  unsigned int j; // eax
  __int64 v43; // rdi
  int v44; // r9d
  int v45; // r9d
  unsigned int v46; // eax
  __int64 v47; // [rsp+8h] [rbp-58h]
  unsigned __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+18h] [rbp-48h]
  __int64 v51; // [rsp+18h] [rbp-48h]
  unsigned int v52; // [rsp+20h] [rbp-40h]
  int v53; // [rsp+24h] [rbp-3Ch]

  result = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( result == a2 + 48 )
    return result;
  if ( !result )
    BUG();
  v4 = result - 24;
  result = (unsigned int)*(unsigned __int8 *)(result - 24) - 30;
  if ( (unsigned int)result > 0xA )
    return result;
  v6 = a2;
  result = sub_B46E30(v4);
  v53 = result;
  if ( !(_DWORD)result )
    return result;
  v7 = 0;
  v48 = (unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32;
  do
  {
    result = sub_B46EC0(v4, v7);
    v8 = result;
    if ( a3 == result )
      goto LABEL_22;
    v9 = *a1;
    v10 = *(_BYTE *)(*a1 + 256) & 1;
    if ( v10 )
    {
      v11 = v9 + 264;
      v12 = 7;
    }
    else
    {
      v18 = *(_DWORD *)(v9 + 272);
      v11 = *(_QWORD *)(v9 + 264);
      if ( !v18 )
      {
        v20 = *(_DWORD *)(v9 + 256);
        ++*(_QWORD *)(v9 + 248);
        v15 = 0;
        v21 = (v20 >> 1) + 1;
LABEL_31:
        v22 = 3 * v18;
        goto LABEL_32;
      }
      v12 = v18 - 1;
    }
    v13 = 1;
    v14 = 0;
    for ( result = v12
                 & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                  * (v48 | ((unsigned int)result >> 9) ^ ((unsigned int)result >> 4))) >> 31)
                  ^ (484763065 * ((unsigned int)v48 | ((unsigned int)result >> 9) ^ ((unsigned int)result >> 4))));
          ;
          result = v12 & v19 )
    {
      v15 = (__int64 *)(v11 + 16LL * (unsigned int)result);
      v16 = *v15;
      if ( v6 == *v15 )
        break;
      if ( v16 == -4096 )
        goto LABEL_28;
LABEL_12:
      if ( v16 == -8192 && v15[1] == -8192 && !v14 )
        v14 = (__int64 *)(v11 + 16LL * (unsigned int)result);
LABEL_29:
      v19 = v13 + result;
      ++v13;
    }
    if ( v8 == v15[1] )
      goto LABEL_22;
    if ( v16 != -4096 )
      goto LABEL_12;
LABEL_28:
    if ( v15[1] != -4096 )
      goto LABEL_29;
    v20 = *(_DWORD *)(v9 + 256);
    if ( v14 )
      v15 = v14;
    ++*(_QWORD *)(v9 + 248);
    v21 = (v20 >> 1) + 1;
    if ( !v10 )
    {
      v18 = *(_DWORD *)(v9 + 272);
      goto LABEL_31;
    }
    v22 = 24;
    v18 = 8;
LABEL_32:
    if ( 4 * v21 >= v22 )
    {
      v50 = v9;
      sub_F19620((const __m128i *)(v9 + 248), 2 * v18);
      v9 = v50;
      if ( (*(_BYTE *)(v50 + 256) & 1) != 0 )
      {
        v32 = v50 + 264;
        v33 = 7;
LABEL_58:
        v34 = 1;
        v35 = 0;
        for ( i = v33
                & (((0xBF58476D1CE4E5B9LL * (v48 | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))) >> 31)
                 ^ (484763065 * (v48 | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; i = v33 & v38 )
        {
          v15 = (__int64 *)(v32 + 16LL * i);
          v37 = *v15;
          if ( v6 == *v15 && v8 == v15[1] )
            break;
          if ( v37 == -4096 )
          {
            if ( v15[1] == -4096 )
            {
LABEL_82:
              if ( v35 )
                v15 = v35;
              break;
            }
          }
          else if ( v37 == -8192 && v15[1] == -8192 && !v35 )
          {
            v35 = (__int64 *)(v32 + 16LL * i);
          }
          v38 = v34 + i;
          ++v34;
        }
LABEL_84:
        v20 = *(_DWORD *)(v9 + 256);
        goto LABEL_34;
      }
      v44 = *(_DWORD *)(v50 + 272);
      v32 = *(_QWORD *)(v50 + 264);
      if ( v44 )
      {
        v33 = v44 - 1;
        goto LABEL_58;
      }
LABEL_97:
      *(_DWORD *)(v9 + 256) = (2 * (*(_DWORD *)(v9 + 256) >> 1) + 2) | *(_DWORD *)(v9 + 256) & 1;
      BUG();
    }
    if ( v18 - *(_DWORD *)(v9 + 260) - v21 <= v18 >> 3 )
    {
      v51 = v9;
      sub_F19620((const __m128i *)(v9 + 248), v18);
      v9 = v51;
      if ( (*(_BYTE *)(v51 + 256) & 1) != 0 )
      {
        v39 = v51 + 264;
        v40 = 7;
      }
      else
      {
        v45 = *(_DWORD *)(v51 + 272);
        v39 = *(_QWORD *)(v51 + 264);
        if ( !v45 )
          goto LABEL_97;
        v40 = v45 - 1;
      }
      v41 = 1;
      v35 = 0;
      for ( j = v40
              & (((0xBF58476D1CE4E5B9LL * (v48 | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4))) >> 31)
               ^ (484763065 * (v48 | ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)))); ; j = v40 & v46 )
      {
        v15 = (__int64 *)(v39 + 16LL * j);
        v43 = *v15;
        if ( v6 == *v15 && v8 == v15[1] )
          break;
        if ( v43 == -4096 )
        {
          if ( v15[1] == -4096 )
            goto LABEL_82;
        }
        else if ( v43 == -8192 && v15[1] == -8192 && !v35 )
        {
          v35 = (__int64 *)(v39 + 16LL * j);
        }
        v46 = v41 + j;
        ++v41;
      }
      goto LABEL_84;
    }
LABEL_34:
    *(_DWORD *)(v9 + 256) = (2 * (v20 >> 1) + 2) | v20 & 1;
    if ( *v15 != -4096 || v15[1] != -4096 )
      --*(_DWORD *)(v9 + 260);
    *v15 = v6;
    v15[1] = v8;
    result = sub_AA5930(v8);
    v49 = v23;
    if ( v23 != result )
    {
      v47 = v4;
      v24 = a1;
      v17 = result;
      v52 = v7;
      v25 = v6;
      do
      {
        v26 = 32LL * (*(_DWORD *)(v17 + 4) & 0x7FFFFFF);
        if ( (*(_BYTE *)(v17 + 7) & 0x40) != 0 )
        {
          v27 = *(_QWORD *)(v17 - 8);
          v28 = v27 + v26;
        }
        else
        {
          v28 = v17;
          v27 = v17 - v26;
        }
        for ( ; v28 != v27; v27 += 32LL )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v17 - 8)
                         + 32LL * *(unsigned int *)(v17 + 72)
                         + 8LL * (unsigned int)((__int64)(v27 - *(_QWORD *)(v17 - 8)) >> 5)) == v25
            && **(_BYTE **)v27 != 13 )
          {
            v29 = sub_ACADE0(*(__int64 ***)(v17 + 8));
            if ( *(_QWORD *)v27 )
            {
              v30 = *(_QWORD *)(v27 + 8);
              **(_QWORD **)(v27 + 16) = v30;
              if ( v30 )
                *(_QWORD *)(v30 + 16) = *(_QWORD *)(v27 + 16);
            }
            *(_QWORD *)v27 = v29;
            if ( v29 )
            {
              v31 = *(_QWORD *)(v29 + 16);
              *(_QWORD *)(v27 + 8) = v31;
              if ( v31 )
                *(_QWORD *)(v31 + 16) = v27 + 8;
              *(_QWORD *)(v27 + 16) = v29 + 16;
              *(_QWORD *)(v29 + 16) = v27;
            }
            *(_BYTE *)v24[1] = 1;
          }
        }
        result = *(_QWORD *)(v17 + 32);
        if ( !result )
          BUG();
        v17 = 0;
        if ( *(_BYTE *)(result - 24) == 84 )
          v17 = result - 24;
      }
      while ( v49 != v17 );
      a1 = v24;
      v6 = v25;
      v4 = v47;
      v7 = v52;
    }
LABEL_22:
    ++v7;
  }
  while ( v53 != v7 );
  return result;
}
