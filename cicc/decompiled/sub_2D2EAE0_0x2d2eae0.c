// Function: sub_2D2EAE0
// Address: 0x2d2eae0
//
_QWORD *__fastcall sub_2D2EAE0(__int64 a1, int a2)
{
  __int64 v3; // r13
  unsigned int v4; // eax
  _QWORD *result; // rax
  __int64 v6; // r9
  __int64 v7; // rdx
  __int64 v8; // r13
  _QWORD *v9; // r14
  _QWORD *i; // rdx
  _QWORD *v11; // rbx
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  __int64 v15; // rdi
  int v16; // r11d
  __int64 *v17; // r10
  unsigned int v18; // ecx
  __int64 *v19; // r12
  __int64 v20; // rsi
  void *v21; // rdi
  unsigned int v22; // r10d
  unsigned __int64 v23; // rdi
  _QWORD *v24; // r11
  const void *v25; // rsi
  size_t v26; // r8
  __int64 v27; // rdx
  _QWORD *j; // rdx
  _QWORD *v29; // [rsp+8h] [rbp-48h]
  _QWORD *v30; // [rsp+8h] [rbp-48h]
  unsigned int v31; // [rsp+14h] [rbp-3Ch]
  unsigned int v32; // [rsp+14h] [rbp-3Ch]
  __int64 v33; // [rsp+18h] [rbp-38h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v33 = *(_QWORD *)(a1 + 8);
  v4 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v4 < 0x40 )
    v4 = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_C7D670((unsigned __int64)v4 << 6, 8);
  v6 = v33;
  *(_QWORD *)(a1 + 8) = result;
  if ( v33 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = v3 << 6;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = (_QWORD *)(v33 + v8);
    for ( i = &result[8 * v7]; i != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = (_QWORD *)(v33 + 24);
    if ( v9 != (_QWORD *)v33 )
    {
      while ( 1 )
      {
        v12 = *(v11 - 3);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *(v11 - 3);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 1;
          v17 = 0;
          v18 = v14 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 6));
          v20 = *v19;
          if ( v12 != *v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = (__int64 *)(v15 + ((unsigned __int64)v18 << 6));
              v20 = *v19;
              if ( v12 == *v19 )
                goto LABEL_15;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_15:
          *v19 = v12;
          v21 = v19 + 3;
          v19[1] = (__int64)(v19 + 3);
          v19[2] = 0x100000000LL;
          v22 = *((_DWORD *)v11 - 2);
          if ( v22 && v19 + 1 != v11 - 2 )
          {
            v24 = (_QWORD *)*(v11 - 2);
            if ( v11 == v24 )
            {
              v25 = v11;
              v26 = 40;
              if ( v22 == 1 )
                goto LABEL_23;
              v30 = (_QWORD *)*(v11 - 2);
              v32 = *((_DWORD *)v11 - 2);
              v36 = v6;
              sub_C8D5F0((__int64)(v19 + 1), v19 + 3, v22, 0x28u, 40, v6);
              v21 = (void *)v19[1];
              v25 = (const void *)*(v11 - 2);
              v6 = v36;
              v22 = v32;
              v24 = v30;
              v26 = 40LL * *((unsigned int *)v11 - 2);
              if ( v26 )
              {
LABEL_23:
                v29 = v24;
                v31 = v22;
                v35 = v6;
                memcpy(v21, v25, v26);
                v6 = v35;
                *((_DWORD *)v19 + 4) = v31;
                *((_DWORD *)v29 - 2) = 0;
              }
              else
              {
                *((_DWORD *)v19 + 4) = v32;
                *((_DWORD *)v30 - 2) = 0;
              }
            }
            else
            {
              v19[1] = (__int64)v24;
              *((_DWORD *)v19 + 4) = *((_DWORD *)v11 - 2);
              *((_DWORD *)v19 + 5) = *((_DWORD *)v11 - 1);
              *(v11 - 2) = v11;
              *((_DWORD *)v11 - 1) = 0;
              *((_DWORD *)v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = *(v11 - 2);
          if ( v11 != (_QWORD *)v23 )
          {
            v34 = v6;
            _libc_free(v23);
            v6 = v34;
          }
        }
        if ( v9 == v11 + 5 )
          break;
        v11 += 8;
      }
    }
    return (_QWORD *)sub_C7D6A0(v6, v8, 8);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[8 * v27]; j != result; result += 8 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
