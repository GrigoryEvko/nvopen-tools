// Function: sub_37BEC40
// Address: 0x37bec40
//
_DWORD *__fastcall sub_37BEC40(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // eax
  _DWORD *result; // rax
  __int64 v7; // r12
  _DWORD *i; // rdx
  __int64 v9; // r15
  unsigned int v10; // ecx
  int v11; // esi
  int v12; // esi
  __int64 v13; // r9
  int v14; // r11d
  __int64 v15; // r8
  __int64 v16; // r10
  __int64 v17; // rdx
  __int64 v18; // r14
  int v19; // edi
  unsigned int v20; // ecx
  __int64 v21; // rcx
  __int64 v22; // rsi
  int v23; // r8d
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  __int64 v27; // rdx
  _DWORD *j; // rdx
  __int64 v29; // [rsp+8h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_DWORD *)sub_C7D670(88LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v29 = 88 * v3;
    v7 = v4 + 88 * v3;
    for ( i = &result[22 * *(unsigned int *)(a1 + 24)]; i != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        while ( 1 )
        {
          v10 = *(_DWORD *)v9;
          if ( *(_DWORD *)v9 <= 0xFFFFFFFD )
          {
            v11 = *(_DWORD *)(a1 + 24);
            if ( !v11 )
            {
              MEMORY[0] = *(_DWORD *)v9;
              BUG();
            }
            v12 = v11 - 1;
            v13 = *(_QWORD *)(a1 + 8);
            v14 = 1;
            v15 = v12 & v10;
            v16 = 0;
            v17 = 11 * v15;
            v18 = v13 + 88 * v15;
            v19 = *(_DWORD *)v18;
            if ( v10 != *(_DWORD *)v18 )
            {
              while ( v19 != -1 )
              {
                if ( !v16 && v19 == -2 )
                  v16 = v18;
                v15 = v12 & (unsigned int)(v14 + v15);
                v17 = 11LL * (unsigned int)v15;
                v18 = v13 + 88LL * (unsigned int)v15;
                v19 = *(_DWORD *)v18;
                if ( v10 == *(_DWORD *)v18 )
                  goto LABEL_14;
                ++v14;
              }
              if ( v16 )
                v18 = v16;
            }
LABEL_14:
            v20 = *(_DWORD *)v9;
            *(_QWORD *)(v18 + 16) = 0x400000000LL;
            *(_DWORD *)v18 = v20;
            *(_QWORD *)(v18 + 8) = v18 + 24;
            if ( *(_DWORD *)(v9 + 16) )
              sub_37B6F50(v18 + 8, (char **)(v9 + 8), v17, v18 + 24, v15, v13);
            v21 = *(_QWORD *)(v9 + 56);
            v22 = v18 + 48;
            if ( v21 )
            {
              v23 = *(_DWORD *)(v9 + 48);
              *(_QWORD *)(v18 + 56) = v21;
              *(_DWORD *)(v18 + 48) = v23;
              *(_QWORD *)(v18 + 64) = *(_QWORD *)(v9 + 64);
              *(_QWORD *)(v18 + 72) = *(_QWORD *)(v9 + 72);
              *(_QWORD *)(v21 + 8) = v22;
              *(_QWORD *)(v18 + 80) = *(_QWORD *)(v9 + 80);
              *(_QWORD *)(v9 + 56) = 0;
              *(_QWORD *)(v9 + 64) = v9 + 48;
              *(_QWORD *)(v9 + 72) = v9 + 48;
              *(_QWORD *)(v9 + 80) = 0;
            }
            else
            {
              *(_DWORD *)(v18 + 48) = 0;
              *(_QWORD *)(v18 + 56) = 0;
              *(_QWORD *)(v18 + 64) = v22;
              *(_QWORD *)(v18 + 72) = v22;
              *(_QWORD *)(v18 + 80) = 0;
            }
            ++*(_DWORD *)(a1 + 16);
            v24 = *(_QWORD *)(v9 + 56);
            while ( v24 )
            {
              sub_37B80B0(*(_QWORD *)(v24 + 24));
              v25 = v24;
              v24 = *(_QWORD *)(v24 + 16);
              j_j___libc_free_0(v25);
            }
            v26 = *(_QWORD *)(v9 + 8);
            if ( v26 != v9 + 24 )
              break;
          }
          v9 += 88;
          if ( v7 == v9 )
            return (_DWORD *)sub_C7D6A0(v4, v29, 8);
        }
        _libc_free(v26);
        v9 += 88;
      }
      while ( v7 != v9 );
    }
    return (_DWORD *)sub_C7D6A0(v4, v29, 8);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[22 * v27]; j != result; result += 22 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
