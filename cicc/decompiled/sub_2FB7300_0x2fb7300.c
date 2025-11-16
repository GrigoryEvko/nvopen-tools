// Function: sub_2FB7300
// Address: 0x2fb7300
//
_DWORD *__fastcall sub_2FB7300(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _DWORD *result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  _DWORD *v11; // rdx
  __int64 i; // r15
  __int64 v13; // r13
  __int64 v14; // rbx
  unsigned int v15; // eax
  int v16; // edx
  int v17; // edx
  __int64 v18; // rdi
  int v19; // r11d
  int *v20; // r10
  unsigned int v21; // ecx
  int *v22; // r12
  int v23; // esi
  __int64 v24; // r10
  __int64 v25; // rax
  unsigned int v26; // r11d
  unsigned int v27; // r11d
  unsigned __int64 v28; // rdi
  int v29; // r11d
  size_t v30; // rdx
  int v31; // r11d
  size_t v32; // rdx
  __int64 v33; // rdx
  _DWORD *j; // rdx
  __int64 v35; // [rsp+8h] [rbp-58h]
  int v36; // [rsp+14h] [rbp-4Ch]
  int v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v39 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_DWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v10 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v38 = 72 * v4;
    v11 = &result[18 * v10];
    for ( i = 72 * v4 + v5; v11 != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
    v13 = v39 + 72;
    v14 = v39 + 56;
    if ( i != v39 )
    {
      while ( 1 )
      {
        v15 = *(_DWORD *)(v14 - 56);
        if ( v15 <= 0xFFFFFFFD )
        {
          v16 = *(_DWORD *)(a1 + 24);
          if ( !v16 )
          {
            MEMORY[0] = *(_DWORD *)(v14 - 56);
            BUG();
          }
          v17 = v16 - 1;
          v18 = *(_QWORD *)(a1 + 8);
          v19 = 1;
          v20 = 0;
          v21 = v17 & (37 * v15);
          v22 = (int *)(v18 + 72LL * v21);
          v23 = *v22;
          if ( v15 != *v22 )
          {
            while ( v23 != -1 )
            {
              if ( !v20 && v23 == -2 )
                v20 = v22;
              v8 = (unsigned int)(v19 + 1);
              v21 = v17 & (v19 + v21);
              v22 = (int *)(v18 + 72LL * v21);
              v23 = *v22;
              if ( v15 == *v22 )
                goto LABEL_15;
              ++v19;
            }
            if ( v20 )
              v22 = v20;
          }
LABEL_15:
          v24 = (__int64)(v22 + 14);
          *v22 = *(_DWORD *)(v14 - 56);
          *((_QWORD *)v22 + 1) = *(_QWORD *)(v14 - 48);
          *((_QWORD *)v22 + 2) = *(_QWORD *)(v14 - 40);
          *((_QWORD *)v22 + 3) = *(_QWORD *)(v14 - 32);
          v25 = *(_QWORD *)(v14 - 24);
          *((_QWORD *)v22 + 5) = v22 + 14;
          *((_QWORD *)v22 + 4) = v25;
          *((_QWORD *)v22 + 6) = 0;
          v26 = *(_DWORD *)(v14 - 8);
          if ( v26 && v22 + 10 != (int *)(v14 - 16) )
          {
            v8 = *(_QWORD *)(v14 - 16);
            if ( v14 == v8 )
            {
              v35 = *(_QWORD *)(v14 - 16);
              v36 = *(_DWORD *)(v14 - 8);
              sub_C8D5F0((__int64)(v22 + 10), v22 + 14, v26, 8u, v8, v9);
              v24 = (__int64)(v22 + 14);
              v31 = v36;
              v8 = v35;
              v32 = 8LL * *(unsigned int *)(v14 - 8);
              if ( v32 )
              {
                memcpy(*((void **)v22 + 5), *(const void **)(v14 - 16), v32);
                v8 = v35;
                v31 = v36;
                v24 = (__int64)(v22 + 14);
              }
              v22[12] = v31;
              *(_DWORD *)(v8 - 8) = 0;
            }
            else
            {
              *((_QWORD *)v22 + 5) = v8;
              v22[12] = *(_DWORD *)(v14 - 8);
              v22[13] = *(_DWORD *)(v14 - 4);
              *(_QWORD *)(v14 - 16) = v14;
              *(_DWORD *)(v14 - 4) = 0;
              *(_DWORD *)(v14 - 8) = 0;
            }
          }
          *((_QWORD *)v22 + 8) = 0;
          *((_QWORD *)v22 + 7) = v22 + 18;
          v27 = *(_DWORD *)(v14 + 8);
          if ( v27 && v14 != v24 )
          {
            if ( v13 == *(_QWORD *)v14 )
            {
              v37 = *(_DWORD *)(v14 + 8);
              sub_C8D5F0(v24, v22 + 18, v27, 8u, v8, v9);
              v29 = v37;
              v30 = 8LL * *(unsigned int *)(v14 + 8);
              if ( v30 )
              {
                memcpy(*((void **)v22 + 7), *(const void **)v14, v30);
                v29 = v37;
              }
              v22[16] = v29;
              *(_DWORD *)(v14 + 8) = 0;
            }
            else
            {
              *((_QWORD *)v22 + 7) = *(_QWORD *)v14;
              v22[16] = *(_DWORD *)(v14 + 8);
              v22[17] = *(_DWORD *)(v14 + 12);
              *(_QWORD *)v14 = v13;
              *(_DWORD *)(v14 + 12) = 0;
              *(_DWORD *)(v14 + 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          if ( v13 != *(_QWORD *)v14 )
            _libc_free(*(_QWORD *)v14);
          v28 = *(_QWORD *)(v14 - 16);
          if ( v28 != v14 )
            _libc_free(v28);
        }
        v13 += 72;
        if ( i == v14 + 16 )
          break;
        v14 += 72;
      }
    }
    return (_DWORD *)sub_C7D6A0(v39, v38, 8);
  }
  else
  {
    v33 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[18 * v33]; j != result; result += 18 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
