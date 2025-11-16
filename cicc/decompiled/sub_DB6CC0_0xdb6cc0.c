// Function: sub_DB6CC0
// Address: 0xdb6cc0
//
_QWORD *__fastcall sub_DB6CC0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 *v5; // rbx
  bool v6; // zf
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *i; // rdx
  __int64 v10; // r8
  int v11; // edi
  int v12; // r10d
  __int64 *v13; // r9
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rdx
  int v20; // eax

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD **)(a1 + 16);
    v8 = 7LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = (_QWORD *)(a1 + 16);
    v8 = 28;
  }
  for ( i = &result[v8]; i != result; result += 7 )
  {
    if ( result )
      *result = -4096;
  }
  if ( a2 != a3 )
  {
    do
    {
      v19 = *v5;
      if ( *v5 != -4096 && v19 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v10 = a1 + 16;
          v11 = 3;
        }
        else
        {
          v20 = *(_DWORD *)(a1 + 24);
          v10 = *(_QWORD *)(a1 + 16);
          if ( !v20 )
          {
            MEMORY[0] = *v5;
            BUG();
          }
          v11 = v20 - 1;
        }
        v12 = 1;
        v13 = 0;
        v14 = v11 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
        v15 = (__int64 *)(v10 + 56LL * v14);
        v16 = *v15;
        if ( v19 != *v15 )
        {
          while ( v16 != -4096 )
          {
            if ( v16 == -8192 && !v13 )
              v13 = v15;
            v14 = v11 & (v12 + v14);
            v15 = (__int64 *)(v10 + 56LL * v14);
            v16 = *v15;
            if ( v19 == *v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        v15[3] = 0;
        v15[2] = 0;
        *((_DWORD *)v15 + 8) = 0;
        *v15 = v19;
        v15[1] = 1;
        v17 = v5[2];
        ++v5[1];
        v18 = v15[2];
        v15[2] = v17;
        LODWORD(v17) = *((_DWORD *)v5 + 6);
        v5[2] = v18;
        LODWORD(v18) = *((_DWORD *)v15 + 6);
        *((_DWORD *)v15 + 6) = v17;
        LODWORD(v17) = *((_DWORD *)v5 + 7);
        *((_DWORD *)v5 + 6) = v18;
        LODWORD(v18) = *((_DWORD *)v15 + 7);
        *((_DWORD *)v15 + 7) = v17;
        LODWORD(v17) = *((_DWORD *)v5 + 8);
        *((_DWORD *)v5 + 7) = v18;
        LODWORD(v18) = *((_DWORD *)v15 + 8);
        *((_DWORD *)v15 + 8) = v17;
        *((_DWORD *)v5 + 8) = v18;
        *((_BYTE *)v15 + 40) = *((_BYTE *)v5 + 40);
        *((_BYTE *)v15 + 41) = *((_BYTE *)v5 + 41);
        v15[6] = v5[6];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        result = (_QWORD *)sub_C7D6A0(v5[2], 16LL * *((unsigned int *)v5 + 8), 8);
      }
      v5 += 7;
    }
    while ( a3 != v5 );
  }
  return result;
}
