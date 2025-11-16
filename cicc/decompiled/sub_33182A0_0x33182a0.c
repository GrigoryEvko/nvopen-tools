// Function: sub_33182A0
// Address: 0x33182a0
//
__int64 __fastcall sub_33182A0(__int64 a1)
{
  unsigned __int64 *v1; // rbx
  __int64 result; // rax
  __int64 *v3; // r15
  int v5; // esi
  unsigned __int64 v6; // rcx
  int v7; // edx
  int v8; // r11d
  int v9; // r9d
  __int64 v10; // r10
  unsigned __int64 *v11; // rdi
  unsigned __int64 **v12; // r8
  unsigned __int64 *v13; // r13
  unsigned int v14; // eax
  int v15; // r9d
  int v16; // edx
  int v17; // r9d
  int v18; // r11d
  int v19; // esi
  __int64 v20; // r8
  unsigned __int64 *v21; // r10
  unsigned int j; // eax
  unsigned int v23; // eax
  int v24; // edx
  int v25; // esi
  int v26; // eax
  int v27; // r9d
  int v28; // edx
  int v29; // r9d
  int v30; // r10d
  int v31; // esi
  __int64 v32; // r8
  unsigned int i; // eax
  unsigned int v34; // eax
  int v35; // r13d
  int v36; // r11d
  unsigned int v37; // [rsp+8h] [rbp-38h]
  int v38; // [rsp+Ch] [rbp-34h]

  v1 = *(unsigned __int64 **)(a1 + 32);
  result = 3LL * *(unsigned int *)(a1 + 40);
  v3 = (__int64 *)&v1[3 * *(unsigned int *)(a1 + 40)];
  if ( v3 == (__int64 *)v1 )
    return result;
  do
  {
    v5 = *(_DWORD *)(a1 + 24);
    if ( !v5 )
    {
      ++*(_QWORD *)a1;
      goto LABEL_8;
    }
    v6 = *v1;
    v7 = *((_DWORD *)v1 + 4);
    v37 = *(_DWORD *)(a1 + 24);
    v8 = v5 - 1;
    v9 = *((_DWORD *)v1 + 2);
    v10 = *(_QWORD *)(a1 + 8);
    v38 = 1;
    v11 = 0;
    for ( result = (v5 - 1)
                 & ((unsigned int)((0xBF58476D1CE4E5B9LL
                                  * ((unsigned int)(37 * v7)
                                   | ((unsigned __int64)(v9 + ((unsigned int)(v6 >> 9) ^ (unsigned int)(v6 >> 4))) << 32))) >> 31)
                  ^ (756364221 * v7)); ; result = v8 & v14 )
    {
      v12 = (unsigned __int64 **)(v10 + 24LL * (unsigned int)result);
      v13 = *v12;
      if ( *v12 != (unsigned __int64 *)v6 || v9 != *((_DWORD *)v12 + 2) )
        break;
      if ( v7 == *((_DWORD *)v12 + 4) )
        goto LABEL_19;
      if ( !v13 )
        goto LABEL_24;
LABEL_6:
      v14 = v38 + result;
      ++v38;
    }
    if ( v13 )
      goto LABEL_6;
LABEL_24:
    v25 = *((_DWORD *)v12 + 2);
    if ( v25 != -1 )
    {
      if ( v25 == -2 && *((_DWORD *)v12 + 4) == 0x80000000 && !v11 )
        v11 = (unsigned __int64 *)(v10 + 24LL * (unsigned int)result);
      goto LABEL_6;
    }
    if ( *((_DWORD *)v12 + 4) != 0x7FFFFFFF )
      goto LABEL_6;
    v26 = *(_DWORD *)(a1 + 16);
    v5 = *(_DWORD *)(a1 + 24);
    if ( !v11 )
      v11 = (unsigned __int64 *)v12;
    ++*(_QWORD *)a1;
    v24 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v37 )
    {
      if ( v37 - *(_DWORD *)(a1 + 20) - v24 <= v37 >> 3 )
      {
        sub_3317BD0(a1, v37);
        v27 = *(_DWORD *)(a1 + 24);
        v11 = 0;
        if ( v27 )
        {
          v28 = *((_DWORD *)v1 + 4);
          v29 = v27 - 1;
          v30 = 1;
          v31 = *((_DWORD *)v1 + 2);
          v32 = *(_QWORD *)(a1 + 8);
          for ( i = v29
                  & (((0xBF58476D1CE4E5B9LL
                     * ((unsigned int)(37 * v28)
                      | ((unsigned __int64)(v31 + ((unsigned int)(*v1 >> 9) ^ (unsigned int)(*v1 >> 4))) << 32))) >> 31)
                   ^ (756364221 * v28)); ; i = v29 & v34 )
          {
            v11 = (unsigned __int64 *)(v32 + 24LL * i);
            if ( *v11 == *v1 && v31 == *((_DWORD *)v11 + 2) && v28 == *((_DWORD *)v11 + 4) )
              break;
            if ( !*v11 )
            {
              v36 = *((_DWORD *)v11 + 2);
              if ( v36 == -1 )
              {
                if ( *((_DWORD *)v11 + 4) == 0x7FFFFFFF )
                {
                  if ( v13 )
                    v11 = v13;
                  goto LABEL_15;
                }
              }
              else if ( v36 == -2 && *((_DWORD *)v11 + 4) == 0x80000000 && !v13 )
              {
                v13 = (unsigned __int64 *)(v32 + 24LL * i);
              }
            }
            v34 = v30 + i;
            ++v30;
          }
        }
        goto LABEL_15;
      }
      goto LABEL_16;
    }
LABEL_8:
    sub_3317BD0(a1, 2 * v5);
    v15 = *(_DWORD *)(a1 + 24);
    v11 = 0;
    if ( v15 )
    {
      v16 = *((_DWORD *)v1 + 4);
      v17 = v15 - 1;
      v18 = 1;
      v19 = *((_DWORD *)v1 + 2);
      v20 = *(_QWORD *)(a1 + 8);
      v21 = 0;
      for ( j = v17
              & (((0xBF58476D1CE4E5B9LL
                 * ((unsigned int)(37 * v16)
                  | ((unsigned __int64)(v19 + ((unsigned int)(*v1 >> 9) ^ (unsigned int)(*v1 >> 4))) << 32))) >> 31)
               ^ (756364221 * v16)); ; j = v17 & v23 )
      {
        v11 = (unsigned __int64 *)(v20 + 24LL * j);
        if ( *v11 == *v1 && v19 == *((_DWORD *)v11 + 2) && v16 == *((_DWORD *)v11 + 4) )
          break;
        if ( !*v11 )
        {
          v35 = *((_DWORD *)v11 + 2);
          if ( v35 == -1 )
          {
            if ( *((_DWORD *)v11 + 4) == 0x7FFFFFFF )
            {
              if ( v21 )
                v11 = v21;
              break;
            }
          }
          else if ( v35 == -2 && *((_DWORD *)v11 + 4) == 0x80000000 && !v21 )
          {
            v21 = (unsigned __int64 *)(v20 + 24LL * j);
          }
        }
        v23 = v18 + j;
        ++v18;
      }
    }
LABEL_15:
    v24 = *(_DWORD *)(a1 + 16) + 1;
LABEL_16:
    *(_DWORD *)(a1 + 16) = v24;
    if ( *v11 || *((_DWORD *)v11 + 2) != -1 || *((_DWORD *)v11 + 4) != 0x7FFFFFFF )
      --*(_DWORD *)(a1 + 20);
    *v11 = *v1;
    *((_DWORD *)v11 + 2) = *((_DWORD *)v1 + 2);
    result = *((unsigned int *)v1 + 4);
    *((_DWORD *)v11 + 4) = result;
LABEL_19:
    v1 += 3;
  }
  while ( v3 != (__int64 *)v1 );
  return result;
}
