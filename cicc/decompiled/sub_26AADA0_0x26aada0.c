// Function: sub_26AADA0
// Address: 0x26aada0
//
_BYTE *__fastcall sub_26AADA0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r12
  unsigned int v6; // edi
  _BYTE *result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  unsigned __int8 *v10; // r8
  _BYTE *i; // rdx
  unsigned __int8 *v12; // rax
  unsigned __int8 v13; // dl
  int v14; // ecx
  int v15; // edi
  __int64 v16; // r10
  int v17; // r15d
  unsigned __int8 *v18; // rbx
  unsigned int v19; // r9d
  unsigned __int8 *v20; // rcx
  unsigned __int8 v21; // r11
  __int64 v22; // rdx
  _BYTE *j; // rdx

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
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
  result = (_BYTE *)sub_C7D670(v6, 1);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    v9 = v4;
    *(_QWORD *)(a1 + 16) = 0;
    v10 = (unsigned __int8 *)(v5 + v4);
    for ( i = &result[v8]; i != result; ++result )
    {
      if ( result )
        *result = -1;
    }
    if ( v10 != (unsigned __int8 *)v5 )
    {
      v12 = (unsigned __int8 *)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( *v12 <= 0xFDu )
            break;
          if ( v10 == ++v12 )
            return (_BYTE *)sub_C7D6A0(v5, v9, 1);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = 1;
        v18 = 0;
        v19 = (v14 - 1) & (37 * v13);
        v20 = (unsigned __int8 *)(v16 + v19);
        v21 = *v20;
        if ( v13 != *v20 )
        {
          while ( v21 != 0xFF )
          {
            if ( !v18 && v21 == 0xFE )
              v18 = v20;
            v19 = v15 & (v17 + v19);
            v20 = (unsigned __int8 *)(v16 + v19);
            v21 = *v20;
            if ( v13 == *v20 )
              goto LABEL_14;
            ++v17;
          }
          if ( v18 )
            v20 = v18;
        }
LABEL_14:
        ++v12;
        *v20 = v13;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_BYTE *)sub_C7D6A0(v5, v9, 1);
  }
  else
  {
    v22 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[v22]; j != result; ++result )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
