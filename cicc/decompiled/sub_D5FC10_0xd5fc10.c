// Function: sub_D5FC10
// Address: 0xd5fc10
//
__int64 __fastcall sub_D5FC10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
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
  int v17; // eax
  __int64 v18; // rdi
  __int64 v19; // rdi
  int v20; // edx

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
      *(_QWORD *)result = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -4096 && result != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 7;
        }
        else
        {
          v20 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v20 )
          {
            MEMORY[0] = *(_QWORD *)v5;
            BUG();
          }
          v11 = v20 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
        v15 = (__int64 *)(v10 + 40LL * v14);
        v16 = *v15;
        if ( result != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
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
        *((_DWORD *)v15 + 4) = *(_DWORD *)(v5 + 16);
        v15[1] = *(_QWORD *)(v5 + 8);
        v17 = *(_DWORD *)(v5 + 32);
        *(_DWORD *)(v5 + 16) = 0;
        *((_DWORD *)v15 + 8) = v17;
        v15[3] = *(_QWORD *)(v5 + 24);
        *(_DWORD *)(v5 + 32) = 0;
        result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        *(_DWORD *)(a1 + 8) = result;
        if ( *(_DWORD *)(v5 + 32) > 0x40u )
        {
          v18 = *(_QWORD *)(v5 + 24);
          if ( v18 )
            result = j_j___libc_free_0_0(v18);
        }
        if ( *(_DWORD *)(v5 + 16) > 0x40u )
        {
          v19 = *(_QWORD *)(v5 + 8);
          if ( v19 )
            result = j_j___libc_free_0_0(v19);
        }
      }
      v5 += 40;
    }
    while ( a3 != v5 );
  }
  return result;
}
