// Function: sub_1C5FBB0
// Address: 0x1c5fbb0
//
_QWORD *__fastcall sub_1C5FBB0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r13
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 *v7; // r14
  _QWORD *i; // rdx
  __int64 *v9; // rbx
  __int64 v10; // rdx
  int v11; // eax
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *j; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
  v5 = sub_1454B60((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(40LL * v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[5 * v3];
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
    if ( v7 != v4 )
    {
      v9 = v4;
      do
      {
        v10 = *v9;
        if ( *v9 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *v9;
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 40LL * v16);
          v18 = *v17;
          if ( *v17 != v10 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 40LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_14;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_14:
          v17[3] = 0;
          v17[2] = 0;
          *((_DWORD *)v17 + 8) = 0;
          *v17 = v10;
          v17[1] = 1;
          v19 = v9[2];
          ++v9[1];
          v20 = v17[2];
          v17[2] = v19;
          LODWORD(v19) = *((_DWORD *)v9 + 6);
          v9[2] = v20;
          LODWORD(v20) = *((_DWORD *)v17 + 6);
          *((_DWORD *)v17 + 6) = v19;
          LODWORD(v19) = *((_DWORD *)v9 + 7);
          *((_DWORD *)v9 + 6) = v20;
          LODWORD(v20) = *((_DWORD *)v17 + 7);
          *((_DWORD *)v17 + 7) = v19;
          LODWORD(v19) = *((_DWORD *)v9 + 8);
          *((_DWORD *)v9 + 7) = v20;
          LODWORD(v20) = *((_DWORD *)v17 + 8);
          *((_DWORD *)v17 + 8) = v19;
          *((_DWORD *)v9 + 8) = v20;
          ++*(_DWORD *)(a1 + 16);
          j___libc_free_0(v9[2]);
        }
        v9 += 5;
      }
      while ( v7 != v9 );
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * *(unsigned int *)(a1 + 24)]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
