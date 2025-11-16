// Function: sub_19E4130
// Address: 0x19e4130
//
_QWORD *__fastcall sub_19E4130(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 *v4; // r12
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // rsi
  _QWORD *i; // rdx
  __int64 *v10; // rax
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // r8d
  __int64 v14; // r9
  int v15; // r14d
  _QWORD *v16; // r11
  unsigned int v17; // edi
  _QWORD *v18; // rcx
  __int64 v19; // r10
  __int64 v20; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(16LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[2 * v3];
    for ( i = &result[2 * v7]; i != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
    if ( v8 != v4 )
    {
      v10 = v4;
      do
      {
        v11 = *v10;
        if ( *v10 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *v10;
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (_QWORD *)(v14 + 16LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( !v16 && v19 == -16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (_QWORD *)(v14 + 16LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_14;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_14:
          *v18 = v11;
          v18[1] = v10[1];
          ++*(_DWORD *)(a1 + 16);
        }
        v10 += 2;
      }
      while ( v8 != v10 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[2 * v20]; j != result; result += 2 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
