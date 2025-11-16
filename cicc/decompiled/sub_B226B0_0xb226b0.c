// Function: sub_B226B0
// Address: 0xb226b0
//
_QWORD *__fastcall sub_B226B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // rdi
  __int64 v11; // rsi
  int v12; // r9d
  _QWORD *v13; // r8
  unsigned int v14; // edx
  _QWORD *v15; // r12
  _QWORD *v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rdi
  int v19; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD **)(a1 + 16);
    v8 = 9LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v8 = 36;
  }
  for ( i = &result[v8]; i != result; result += 9 )
  {
    if ( result )
      *result = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD **)v5;
      if ( *(_QWORD *)v5 != -8192 && result != (_QWORD *)-4096LL )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 3;
        }
        else
        {
          v19 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v19 )
          {
            MEMORY[0] = *(_QWORD *)v5;
            BUG();
          }
          v11 = (unsigned int)(v19 - 1);
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v15 = (_QWORD *)(v10 + 72LL * v14);
        v16 = (_QWORD *)*v15;
        if ( result != (_QWORD *)*v15 )
        {
          while ( v16 != (_QWORD *)-4096LL )
          {
            if ( v16 == (_QWORD *)-8192LL && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (_QWORD *)(v10 + 72LL * v14);
            v16 = (_QWORD *)*v15;
            if ( result == (_QWORD *)*v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        *v15 = result;
        v15[1] = v15 + 3;
        v15[2] = 0x200000000LL;
        if ( *(_DWORD *)(v5 + 16) )
        {
          v11 = v5 + 8;
          sub_B187A0((__int64)(v15 + 1), (char **)(v5 + 8));
        }
        v15[6] = 0x200000000LL;
        v15[5] = v15 + 7;
        if ( *(_DWORD *)(v5 + 48) )
        {
          v11 = v5 + 40;
          sub_B187A0((__int64)(v15 + 5), (char **)(v5 + 40));
        }
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v17 = *(_QWORD *)(v5 + 40);
        if ( v17 != v5 + 56 )
          _libc_free(v17, v11);
        v18 = *(_QWORD *)(v5 + 8);
        result = (_QWORD *)(v5 + 24);
        if ( v18 != v5 + 24 )
          result = (_QWORD *)_libc_free(v18, v11);
      }
      v5 += 72;
    }
    while ( a3 != v5 );
  }
  return result;
}
