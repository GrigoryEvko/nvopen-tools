// Function: sub_1A11440
// Address: 0x1a11440
//
__int64 *__fastcall sub_1A11440(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  int v8; // r10d
  __int64 v9; // rdx
  __int64 *v10; // r14
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  unsigned int i; // r8d
  __int64 *v15; // r15
  __int64 v16; // rcx
  unsigned int v17; // r8d
  __int64 *v18; // r15
  int v20; // edx
  int v21; // ecx
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rdx
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // rdx
  int v28; // r8d
  __int64 *v29; // rdi
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rsi
  unsigned int j; // eax
  __int64 v33; // rsi
  unsigned int v34; // eax
  int v35; // edx
  int v36; // edx
  __int64 v37; // rsi
  int v38; // r8d
  unsigned int k; // eax
  __int64 v40; // rcx
  unsigned int v41; // eax
  int v42; // [rsp+8h] [rbp-38h]

  v6 = a1 + 184;
  v7 = *(_DWORD *)(a1 + 208);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 184);
    goto LABEL_33;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 192);
  v10 = 0;
  v11 = ((((37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(37 * a3) << 32)) >> 22)
      ^ (((37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(37 * a3) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  v13 = ((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - (v12 << 27));
  for ( i = v13 & (v7 - 1); ; i = (v7 - 1) & v17 )
  {
    v15 = (__int64 *)(v9 + 24LL * i);
    v16 = *v15;
    if ( a2 == *v15 && a3 == *((_DWORD *)v15 + 2) )
      return v15 + 2;
    if ( v16 == -8 )
      break;
    if ( v16 == -16 && *((_DWORD *)v15 + 2) == -2 && !v10 )
      v10 = (__int64 *)(v9 + 24LL * i);
LABEL_9:
    v17 = v8 + i;
    ++v8;
  }
  if ( *((_DWORD *)v15 + 2) != -1 )
    goto LABEL_9;
  v20 = *(_DWORD *)(a1 + 200);
  if ( !v10 )
    v10 = v15;
  ++*(_QWORD *)(a1 + 184);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v7 )
  {
LABEL_33:
    sub_1A10020(v6, 2 * v7);
    v25 = *(_DWORD *)(a1 + 208);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 192);
      v28 = 1;
      v29 = 0;
      v30 = ((((37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(37 * a3) << 32)) >> 22)
          ^ (((37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(37 * a3) << 32));
      v31 = ((9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13)))) >> 15)
          ^ (9 * (((v30 - 1 - (v30 << 13)) >> 8) ^ (v30 - 1 - (v30 << 13))));
      for ( j = v26 & (((v31 - 1 - (v31 << 27)) >> 31) ^ (v31 - 1 - ((_DWORD)v31 << 27))); ; j = v26 & v34 )
      {
        v10 = (__int64 *)(v27 + 24LL * j);
        v33 = *v10;
        if ( a2 == *v10 && a3 == *((_DWORD *)v10 + 2) )
          break;
        if ( v33 == -8 )
        {
          if ( *((_DWORD *)v10 + 2) == -1 )
          {
LABEL_56:
            if ( v29 )
              v10 = v29;
            v21 = *(_DWORD *)(a1 + 200) + 1;
            goto LABEL_18;
          }
        }
        else if ( v33 == -16 && *((_DWORD *)v10 + 2) == -2 && !v29 )
        {
          v29 = (__int64 *)(v27 + 24LL * j);
        }
        v34 = v28 + j;
        ++v28;
      }
      goto LABEL_52;
    }
LABEL_61:
    ++*(_DWORD *)(a1 + 200);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 204) - v21 <= v7 >> 3 )
  {
    v42 = v13;
    sub_1A10020(v6, v7);
    v35 = *(_DWORD *)(a1 + 208);
    if ( v35 )
    {
      v36 = v35 - 1;
      v29 = 0;
      v38 = 1;
      for ( k = v36 & v42; ; k = v36 & v41 )
      {
        v37 = *(_QWORD *)(a1 + 192);
        v10 = (__int64 *)(v37 + 24LL * k);
        v40 = *v10;
        if ( a2 == *v10 && a3 == *((_DWORD *)v10 + 2) )
          break;
        if ( v40 == -8 )
        {
          if ( *((_DWORD *)v10 + 2) == -1 )
            goto LABEL_56;
        }
        else if ( v40 == -16 && *((_DWORD *)v10 + 2) == -2 && !v29 )
        {
          v29 = (__int64 *)(v37 + 24LL * k);
        }
        v41 = v38 + k;
        ++v38;
      }
LABEL_52:
      v21 = *(_DWORD *)(a1 + 200) + 1;
      goto LABEL_18;
    }
    goto LABEL_61;
  }
LABEL_18:
  *(_DWORD *)(a1 + 200) = v21;
  if ( *v10 != -8 || *((_DWORD *)v10 + 2) != -1 )
    --*(_DWORD *)(a1 + 204);
  *v10 = a2;
  v18 = v10 + 2;
  *((_DWORD *)v10 + 2) = a3;
  v10[2] = 0;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u )
  {
    v22 = sub_15A0A60(a2, a3);
    if ( v22 )
    {
      if ( *(_BYTE *)(v22 + 16) != 9 )
      {
        v23 = v10[2];
        v24 = (v23 >> 1) & 3;
        if ( v24 != 1 )
        {
          if ( v24 )
          {
            if ( v22 != (v23 & 0xFFFFFFFFFFFFFFF8LL) )
              v10[2] = v23 | 6;
          }
          else
          {
            v10[2] = v23 & 1 | v22 | 2;
          }
        }
      }
    }
    else if ( (((unsigned __int8)v10[2] ^ 6) & 6) != 0 )
    {
      v10[2] |= 6uLL;
    }
  }
  return v18;
}
