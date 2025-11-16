// Function: sub_35D4CD0
// Address: 0x35d4cd0
//
unsigned __int64 __fastcall sub_35D4CD0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v8; // rdi
  unsigned int v9; // esi
  __int64 v10; // rcx
  int v11; // r11d
  unsigned int v12; // r8d
  unsigned __int64 result; // rax
  _QWORD *v14; // rdx
  unsigned int i; // r9d
  _QWORD *v16; // r8
  __int64 v17; // r15
  unsigned int v18; // r9d
  _DWORD *v19; // rdx
  int v20; // ecx
  int v21; // r8d
  int v22; // edx
  __int64 v23; // rcx
  int v24; // r8d
  _QWORD *v25; // r9
  int v26; // edi
  unsigned int j; // eax
  __int64 v28; // rsi
  unsigned int v29; // eax
  int v30; // edx
  int v31; // ecx
  __int64 v32; // rdi
  int v33; // r8d
  unsigned int k; // eax
  __int64 v35; // rsi
  unsigned int v36; // eax
  int v37; // [rsp+8h] [rbp-38h]

  v8 = a1 + 32;
  v9 = *(_DWORD *)(a1 + 56);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_24;
  }
  v10 = *(_QWORD *)(a1 + 40);
  v11 = 1;
  v12 = (unsigned int)a3 >> 9;
  result = ((0xBF58476D1CE4E5B9LL
           * (v12 ^ ((unsigned int)a3 >> 4)
            | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
         ^ (0xBF58476D1CE4E5B9LL
          * (v12 ^ ((unsigned int)a3 >> 4)
           | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32)));
  v14 = 0;
  for ( i = (((0xBF58476D1CE4E5B9LL
             * (v12 ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (484763065 * (v12 ^ ((unsigned int)a3 >> 4))))
          & (v9 - 1); ; i = (v9 - 1) & v18 )
  {
    v16 = (_QWORD *)(v10 + 24LL * i);
    v17 = *v16;
    if ( a2 == *v16 && a3 == v16[1] )
    {
      v19 = v16 + 2;
      goto LABEL_12;
    }
    if ( v17 == -4096 )
      break;
    if ( v17 == -8192 && v16[1] == -8192 && !v14 )
      v14 = (_QWORD *)(v10 + 24LL * i);
LABEL_9:
    v18 = v11 + i;
    ++v11;
  }
  if ( v16[1] != -4096 )
    goto LABEL_9;
  v20 = *(_DWORD *)(a1 + 48);
  if ( !v14 )
    v14 = v16;
  ++*(_QWORD *)(a1 + 32);
  v21 = v20 + 1;
  if ( 4 * (v20 + 1) >= 3 * v9 )
  {
LABEL_24:
    sub_35D4A00(v8, 2 * v9);
    v22 = *(_DWORD *)(a1 + 56);
    if ( v22 )
    {
      v24 = 1;
      v25 = 0;
      v26 = v22 - 1;
      for ( j = (v22 - 1)
              & (((0xBF58476D1CE4E5B9LL
                 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
                  | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
               ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = v26 & v29 )
      {
        v23 = *(_QWORD *)(a1 + 40);
        v14 = (_QWORD *)(v23 + 24LL * j);
        v28 = *v14;
        if ( a2 == *v14 && a3 == v14[1] )
          break;
        if ( v28 == -4096 )
        {
          if ( v14[1] == -4096 )
          {
LABEL_47:
            result = *(unsigned int *)(a1 + 48);
            if ( v25 )
              v14 = v25;
            v21 = result + 1;
            goto LABEL_18;
          }
        }
        else if ( v28 == -8192 && v14[1] == -8192 && !v25 )
        {
          v25 = (_QWORD *)(v23 + 24LL * j);
        }
        v29 = v24 + j;
        ++v24;
      }
      goto LABEL_43;
    }
LABEL_52:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
  if ( v9 - *(_DWORD *)(a1 + 52) - v21 <= v9 >> 3 )
  {
    v37 = result;
    sub_35D4A00(v8, v9);
    v30 = *(_DWORD *)(a1 + 56);
    if ( v30 )
    {
      v31 = v30 - 1;
      v25 = 0;
      v33 = 1;
      for ( k = (v30 - 1) & v37; ; k = v31 & v36 )
      {
        v32 = *(_QWORD *)(a1 + 40);
        v14 = (_QWORD *)(v32 + 24LL * k);
        v35 = *v14;
        if ( a2 == *v14 && a3 == v14[1] )
          break;
        if ( v35 == -4096 )
        {
          if ( v14[1] == -4096 )
            goto LABEL_47;
        }
        else if ( v35 == -8192 && v14[1] == -8192 && !v25 )
        {
          v25 = (_QWORD *)(v32 + 24LL * k);
        }
        v36 = v33 + k;
        ++v33;
      }
LABEL_43:
      result = *(unsigned int *)(a1 + 48);
      v21 = result + 1;
      goto LABEL_18;
    }
    goto LABEL_52;
  }
LABEL_18:
  *(_DWORD *)(a1 + 48) = v21;
  if ( *v14 != -4096 || v14[1] != -4096 )
    --*(_DWORD *)(a1 + 52);
  *v14 = a2;
  v19 = v14 + 2;
  *((_QWORD *)v19 - 1) = a3;
  *v19 = 0;
LABEL_12:
  *v19 = a4;
  return result;
}
