// Function: sub_3317EC0
// Address: 0x3317ec0
//
__int64 __fastcall sub_3317EC0(__int64 a1, __int64 a2, unsigned __int64 *a3)
{
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx
  unsigned __int64 v8; // r8
  int v9; // edi
  int v10; // r10d
  __int64 v11; // r11
  __int64 v12; // rcx
  unsigned int i; // r15d
  __int64 *v14; // r9
  __int64 v15; // r13
  unsigned int v16; // r15d
  int v17; // r10d
  int v18; // esi
  int v19; // r10d
  int v20; // r8d
  __int64 v21; // r9
  int v22; // r14d
  unsigned int k; // edx
  unsigned int v24; // edx
  __int64 v25; // rcx
  char v26; // si
  __int64 v27; // rdi
  int v28; // edi
  int v29; // r12d
  int v30; // r10d
  int v31; // esi
  int v32; // r10d
  int v33; // r14d
  int v34; // r8d
  __int64 v35; // r9
  unsigned int j; // edx
  unsigned int v37; // edx
  int v38; // r11d
  int v39; // r11d
  unsigned __int64 *v40; // [rsp+0h] [rbp-40h]
  int v41; // [rsp+8h] [rbp-38h]
  __int64 v42; // [rsp+8h] [rbp-38h]
  __int64 v43; // [rsp+8h] [rbp-38h]

  result = a1;
  v6 = *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  if ( !(_DWORD)v6 )
  {
    *(_QWORD *)a2 = v7 + 1;
    goto LABEL_7;
  }
  v8 = *a3;
  v9 = *((_DWORD *)a3 + 4);
  v40 = a3;
  v10 = *((_DWORD *)a3 + 2);
  v11 = *(_QWORD *)(a2 + 8);
  v41 = 1;
  v12 = 0;
  for ( i = (v6 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((unsigned int)(37 * v9)
              | ((unsigned __int64)(v10 + ((unsigned int)(v8 >> 9) ^ (unsigned int)(v8 >> 4))) << 32))) >> 31)
           ^ (756364221 * v9)); ; i = (v6 - 1) & v16 )
  {
    v14 = (__int64 *)(v11 + 24LL * i);
    v15 = *v14;
    if ( *v14 == v8 && v10 == *((_DWORD *)v14 + 2) && v9 == *((_DWORD *)v14 + 4) )
    {
      v25 = 3 * v6;
      v26 = 0;
      v27 = v11 + 8 * v25;
      v12 = v11 + 24LL * i;
      goto LABEL_21;
    }
    if ( !v15 )
      break;
LABEL_5:
    v16 = v41 + i;
    ++v41;
  }
  v29 = *((_DWORD *)v14 + 2);
  if ( v29 != -1 )
  {
    if ( v29 == -2 && *((_DWORD *)v14 + 4) == 0x80000000 && !v12 )
      v12 = v11 + 24LL * i;
    goto LABEL_5;
  }
  if ( *((_DWORD *)v14 + 4) != 0x7FFFFFFF )
    goto LABEL_5;
  a3 = v40;
  if ( !v12 )
    v12 = v11 + 24LL * i;
  v28 = *(_DWORD *)(a2 + 16) + 1;
  *(_QWORD *)a2 = v7 + 1;
  if ( 4 * v28 < (unsigned int)(3 * v6) )
  {
    if ( (int)v6 - *(_DWORD *)(a2 + 20) - v28 <= (unsigned int)v6 >> 3 )
    {
      v43 = result;
      sub_3317BD0(a2, v6);
      v30 = *(_DWORD *)(a2 + 24);
      v12 = 0;
      result = v43;
      if ( v30 )
      {
        v31 = *((_DWORD *)v40 + 4);
        v32 = v30 - 1;
        v33 = 1;
        v34 = *((_DWORD *)v40 + 2);
        for ( j = v32
                & (((0xBF58476D1CE4E5B9LL
                   * ((unsigned int)(37 * v31)
                    | ((unsigned __int64)(v34 + ((unsigned int)(*v40 >> 9) ^ (unsigned int)(*v40 >> 4))) << 32))) >> 31)
                 ^ (756364221 * v31)); ; j = v32 & v37 )
        {
          v35 = *(_QWORD *)(a2 + 8);
          v12 = v35 + 24LL * j;
          if ( *(_QWORD *)v12 == *v40 && v34 == *(_DWORD *)(v12 + 8) && v31 == *(_DWORD *)(v12 + 16) )
            break;
          if ( !*(_QWORD *)v12 )
          {
            v39 = *(_DWORD *)(v12 + 8);
            if ( v39 == -1 )
            {
              if ( *(_DWORD *)(v12 + 16) == 0x7FFFFFFF )
                goto LABEL_57;
            }
            else if ( v39 == -2 && *(_DWORD *)(v12 + 16) == 0x80000000 && !v15 )
            {
              v15 = v35 + 24LL * j;
            }
          }
          v37 = v33 + j;
          ++v33;
        }
      }
      goto LABEL_17;
    }
    goto LABEL_18;
  }
LABEL_7:
  v42 = result;
  sub_3317BD0(a2, 2 * v6);
  v17 = *(_DWORD *)(a2 + 24);
  v12 = 0;
  result = v42;
  if ( v17 )
  {
    v18 = *((_DWORD *)a3 + 4);
    v19 = v17 - 1;
    v15 = 0;
    v20 = *((_DWORD *)a3 + 2);
    v22 = 1;
    for ( k = v19
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(37 * v18)
                | ((unsigned __int64)(v20 + ((unsigned int)(*a3 >> 9) ^ (unsigned int)(*a3 >> 4))) << 32))) >> 31)
             ^ (756364221 * v18)); ; k = v19 & v24 )
    {
      v21 = *(_QWORD *)(a2 + 8);
      v12 = v21 + 24LL * k;
      if ( *(_QWORD *)v12 == *a3 && v20 == *(_DWORD *)(v12 + 8) && v18 == *(_DWORD *)(v12 + 16) )
        break;
      if ( !*(_QWORD *)v12 )
      {
        v38 = *(_DWORD *)(v12 + 8);
        if ( v38 == -1 )
        {
          if ( *(_DWORD *)(v12 + 16) == 0x7FFFFFFF )
          {
LABEL_57:
            if ( v15 )
              v12 = v15;
            break;
          }
        }
        else if ( v38 == -2 && *(_DWORD *)(v12 + 16) == 0x80000000 && !v15 )
        {
          v15 = v21 + 24LL * k;
        }
      }
      v24 = v22 + k;
      ++v22;
    }
  }
LABEL_17:
  v28 = *(_DWORD *)(a2 + 16) + 1;
LABEL_18:
  *(_DWORD *)(a2 + 16) = v28;
  if ( *(_QWORD *)v12 || *(_DWORD *)(v12 + 8) != -1 || *(_DWORD *)(v12 + 16) != 0x7FFFFFFF )
    --*(_DWORD *)(a2 + 20);
  *(_QWORD *)v12 = *a3;
  *(_DWORD *)(v12 + 8) = *((_DWORD *)a3 + 2);
  *(_DWORD *)(v12 + 16) = *((_DWORD *)a3 + 4);
  v27 = *(_QWORD *)(a2 + 8) + 24LL * *(unsigned int *)(a2 + 24);
  v7 = *(_QWORD *)a2;
  v26 = 1;
LABEL_21:
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 8) = v7;
  *(_QWORD *)(result + 16) = v12;
  *(_QWORD *)(result + 24) = v27;
  *(_BYTE *)(result + 32) = v26;
  return result;
}
