// Function: sub_335F610
// Address: 0x335f610
//
_QWORD *__fastcall sub_335F610(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r12
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // rdi
  _QWORD *i; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  int v14; // esi
  int v15; // esi
  __int64 v16; // r13
  unsigned int v17; // r10d
  _QWORD *v18; // rcx
  __int64 v19; // r11
  __int64 v20; // rdx
  __int64 v21; // rdx
  _QWORD *j; // rdx
  int v23; // r15d
  _QWORD *v24; // r14
  int v25; // ecx

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
  result = (_QWORD *)sub_C7D670(16LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v9 = 16LL * v4;
    v10 = (__int64 *)(v5 + v9);
    for ( i = &result[2 * v8]; i != result; result += 2 )
    {
      if ( result )
        *result = 0x7FFFFFFFFFFFFFFFLL;
    }
    if ( v10 != (__int64 *)v5 )
    {
      v12 = (__int64 *)v5;
      do
      {
        while ( 1 )
        {
          v13 = *v12;
          if ( (unsigned __int64)(*v12 + 0x7FFFFFFFFFFFFFFFLL) <= 0xFFFFFFFFFFFFFFFDLL )
            break;
          v12 += 2;
          if ( v10 == v12 )
            return (_QWORD *)sub_C7D6A0(v5, v9, 8);
        }
        v14 = *(_DWORD *)(a1 + 24);
        if ( !v14 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 8);
        v17 = v15 & (37 * v13);
        v18 = (_QWORD *)(v16 + 16LL * v17);
        v19 = *v18;
        if ( v13 != *v18 )
        {
          v23 = 1;
          v24 = 0;
          while ( v19 != 0x7FFFFFFFFFFFFFFFLL )
          {
            if ( !v24 && v19 == 0x8000000000000000LL )
              v24 = v18;
            v25 = v23++;
            v17 = v15 & (v25 + v17);
            v18 = (_QWORD *)(v16 + 16LL * v17);
            v19 = *v18;
            if ( v13 == *v18 )
              goto LABEL_14;
          }
          if ( v24 )
            v18 = v24;
        }
LABEL_14:
        *v18 = v13;
        v20 = v12[1];
        v12 += 2;
        v18[1] = v20;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v10 != v12 );
    }
    return (_QWORD *)sub_C7D6A0(v5, v9, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v21]; j != result; result += 2 )
    {
      if ( result )
        *result = 0x7FFFFFFFFFFFFFFFLL;
    }
  }
  return result;
}
