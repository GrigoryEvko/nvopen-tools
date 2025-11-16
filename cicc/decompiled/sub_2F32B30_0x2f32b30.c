// Function: sub_2F32B30
// Address: 0x2f32b30
//
_QWORD *__fastcall sub_2F32B30(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rsi
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v7; // edi
  _QWORD *result; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // r13
  _QWORD *i; // rdx
  __int64 j; // rbx
  unsigned __int64 v16; // rdx
  __int64 v17; // r9
  int v18; // eax
  __int64 v19; // rdi
  __int64 v20; // r11
  int v21; // r9d
  unsigned int k; // r10d
  bool v23; // al
  __int64 v24; // rax
  _QWORD *m; // rdx
  int v26; // r10d
  unsigned int v27; // [rsp+Ch] [rbp-54h]
  __int64 v28; // [rsp+10h] [rbp-50h]
  __int64 *v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  unsigned int v31; // [rsp+28h] [rbp-38h]
  int v32; // [rsp+2Ch] [rbp-34h]
  int v33; // [rsp+2Ch] [rbp-34h]

  v2 = (unsigned int)(a2 - 1);
  v3 = 8;
  v5 = *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v7 < 0x40 )
    v7 = 64;
  *(_DWORD *)(a1 + 24) = v7;
  result = (_QWORD *)sub_C7D670(16LL * v7, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v6 )
  {
    v11 = *(unsigned int *)(a1 + 24);
    v12 = 16 * v5;
    *(_QWORD *)(a1 + 16) = 0;
    v13 = v6 + v12;
    for ( i = &result[2 * v11]; i != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
    for ( j = v6; v13 != j; j += 16 )
    {
      v16 = *(_QWORD *)j - 1LL;
      if ( v16 <= 0xFFFFFFFFFFFFFFFDLL )
      {
        v17 = *(unsigned int *)(a1 + 24);
        v32 = *(_DWORD *)(a1 + 24);
        if ( !(_DWORD)v17 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v30 = *(_QWORD *)(a1 + 8);
        v18 = sub_2E8E920((__int64 *)j, v3, v16, v9, v10, v17);
        v19 = *(_QWORD *)j;
        v20 = 0;
        v10 = 1;
        v21 = v32 - 1;
        for ( k = (v32 - 1) & v18; ; k = v21 & v26 )
        {
          v9 = v30 + 16LL * k;
          v3 = *(_QWORD *)v9;
          if ( (unsigned __int64)(*(_QWORD *)v9 - 1LL) > 0xFFFFFFFFFFFFFFFDLL
            || (unsigned __int64)(v19 - 1) > 0xFFFFFFFFFFFFFFFDLL )
          {
            if ( v19 == v3 )
              goto LABEL_18;
            v24 = *(_QWORD *)v9;
            v3 = v19;
          }
          else
          {
            v27 = v10;
            v28 = v20;
            v29 = (__int64 *)(v30 + 16LL * k);
            v31 = k;
            v33 = v21;
            v23 = sub_2E88AF0(v19, v3, 3u);
            v21 = v33;
            k = v31;
            v9 = (__int64)v29;
            v20 = v28;
            v10 = v27;
            if ( v23 )
            {
              v3 = *(_QWORD *)j;
              goto LABEL_18;
            }
            v24 = *v29;
            v3 = *(_QWORD *)j;
          }
          if ( !v24 )
            break;
          if ( !v20 && v24 == -1 )
            v20 = v9;
          v26 = v10 + k;
          v19 = v3;
          v10 = (unsigned int)(v10 + 1);
        }
        if ( v20 )
          v9 = v20;
LABEL_18:
        *(_QWORD *)v9 = v3;
        *(_DWORD *)(v9 + 8) = *(_DWORD *)(j + 8);
        ++*(_DWORD *)(a1 + 16);
      }
    }
    return (_QWORD *)sub_C7D6A0(v6, v12, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[2 * *(unsigned int *)(a1 + 24)]; m != result; result += 2 )
    {
      if ( result )
        *result = 0;
    }
  }
  return result;
}
