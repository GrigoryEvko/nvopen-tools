// Function: sub_1A338A0
// Address: 0x1a338a0
//
__int64 __fastcall sub_1A338A0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 *v10; // rdx
  __int64 v11; // r10
  int v12; // ecx
  __int64 v13; // rdi
  __int64 v14; // rsi
  int v15; // r9d
  __int64 *v16; // r8
  __int64 v17; // rdi
  int v18; // ecx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 32LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 32;
  }
  for ( i = result + v8; i != result; result += 32 )
  {
    if ( result )
      *(_QWORD *)result = -8;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *v5;
      if ( *v5 != -8 && result != -16 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = (__int64 *)(a1 + 16);
          LODWORD(v11) = 0;
          v12 = 0;
          v13 = a1 + 16;
        }
        else
        {
          v18 = *(_DWORD *)(a1 + 24);
          v13 = *(_QWORD *)(a1 + 16);
          if ( !v18 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v12 = v18 - 1;
          v11 = v12 & (((unsigned int)result >> 4) ^ ((unsigned int)result >> 9));
          v10 = (__int64 *)(v13 + 32 * v11);
        }
        v14 = *v10;
        v15 = 1;
        v16 = 0;
        if ( result != *v10 )
        {
          while ( v14 != -8 )
          {
            if ( v14 == -16 && !v16 )
              v16 = v10;
            v11 = v12 & (unsigned int)(v11 + v15);
            v10 = (__int64 *)(v13 + 32 * v11);
            v14 = *v10;
            if ( result == *v10 )
              goto LABEL_11;
            ++v15;
          }
          if ( v16 )
            v10 = v16;
        }
LABEL_11:
        *v10 = result;
        v10[1] = v5[1];
        v10[2] = v5[2];
        v10[3] = v5[3];
        v5[1] = 0;
        v5[3] = 0;
        v5[2] = 0;
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        v17 = v5[1];
        if ( v17 )
          result = j_j___libc_free_0(v17, v5[3] - v17);
      }
      v5 += 4;
    }
    while ( a3 != v5 );
  }
  return result;
}
