// Function: sub_2659D40
// Address: 0x2659d40
//
_QWORD *__fastcall sub_2659D40(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  _QWORD *i; // rdx
  __int64 v10; // rbx
  int v11; // esi
  int v12; // esi
  unsigned __int64 v13; // rdx
  __int64 v14; // r9
  _QWORD *v15; // r10
  int v16; // r11d
  __int64 v17; // r8
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  _QWORD *j; // rdx
  _QWORD *v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(144LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 144 * v3;
    v8 = v4 + 144 * v3;
    for ( i = &result[18 * v7]; i != result; result += 18 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        while ( 1 )
        {
          if ( (*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF0LL) != 0xFFFFFFFFFFFFFFF0LL )
          {
            v11 = *(_DWORD *)(a1 + 24);
            if ( !v11 )
            {
              MEMORY[0] = *(_QWORD *)v10;
              BUG();
            }
            v12 = v11 - 1;
            v13 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 0;
            v16 = 1;
            v17 = (unsigned int)v13 & v12;
            v18 = (_QWORD *)(v14 + 144 * v17);
            v19 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v13 != v19 )
            {
              while ( v19 != -8 )
              {
                if ( !v15 && v19 == -16 )
                  v15 = v18;
                v17 = v12 & (unsigned int)(v16 + v17);
                v18 = (_QWORD *)(v14 + 144LL * (unsigned int)v17);
                v19 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
                if ( v13 == v19 )
                  goto LABEL_14;
                ++v16;
              }
              if ( v15 )
                v18 = v15;
            }
LABEL_14:
            *v18 = *(_QWORD *)v10;
            v20 = *(_QWORD *)(v10 + 8);
            v18[3] = 0xC00000000LL;
            v18[1] = v20;
            v18[2] = v18 + 4;
            v21 = *(unsigned int *)(v10 + 24);
            if ( (_DWORD)v21 )
            {
              v27 = v18;
              sub_263F700((__int64)(v18 + 2), (char **)(v10 + 16), (__int64)(v18 + 4), v21, v17, v14);
              v18 = v27;
            }
            v18[11] = 0xC00000000LL;
            v18[10] = v18 + 12;
            v22 = *(unsigned int *)(v10 + 88);
            if ( (_DWORD)v22 )
              sub_263F700((__int64)(v18 + 10), (char **)(v10 + 80), v22, v21, v17, v14);
            ++*(_DWORD *)(a1 + 16);
            v23 = *(_QWORD *)(v10 + 80);
            if ( v23 != v10 + 96 )
              _libc_free(v23);
            v24 = *(_QWORD *)(v10 + 16);
            if ( v24 != v10 + 32 )
              break;
          }
          v10 += 144;
          if ( v8 == v10 )
            return (_QWORD *)sub_C7D6A0(v4, v28, 8);
        }
        _libc_free(v24);
        v10 += 144;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)sub_C7D6A0(v4, v28, 8);
  }
  else
  {
    v25 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[18 * v25]; j != result; result += 18 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
