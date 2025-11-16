// Function: sub_1A1B2B0
// Address: 0x1a1b2b0
//
__int64 __fastcall sub_1A1B2B0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  bool v6; // zf
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 i; // rdx
  __int64 v10; // rdi
  int v11; // esi
  int v12; // r10d
  __int64 *v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rdi
  int v18; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 40LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 320;
  }
  for ( i = result + v8; i != result; result += 40 )
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
          v10 = a1 + 16;
          v11 = 7;
        }
        else
        {
          v18 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v18 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v11 = v18 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v15 = (__int64 *)(v10 + 40LL * v14);
        v16 = *v15;
        if ( result != *v15 )
        {
          while ( v16 != -8 )
          {
            if ( v16 == -16 && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (__int64 *)(v10 + 40LL * v14);
            v16 = *v15;
            if ( result == *v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        *v15 = result;
        v15[1] = v5[1];
        v15[2] = v5[2];
        v15[3] = v5[3];
        v15[4] = v5[4];
        v5[2] = 0;
        v5[4] = 0;
        v5[3] = 0;
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        v17 = v5[2];
        if ( v17 )
          result = j_j___libc_free_0(v17, v5[4] - v17);
      }
      v5 += 5;
    }
    while ( a3 != v5 );
  }
  return result;
}
