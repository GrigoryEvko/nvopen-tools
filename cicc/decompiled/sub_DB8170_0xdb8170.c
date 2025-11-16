// Function: sub_DB8170
// Address: 0xdb8170
//
unsigned __int64 __fastcall sub_DB8170(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  bool v6; // zf
  unsigned __int64 result; // rax
  __int64 v8; // rdx
  unsigned __int64 i; // rdx
  __int64 v10; // r8
  __int64 v11; // rsi
  int v12; // r10d
  __int64 v13; // r9
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rcx
  char v17; // al
  __int64 v18; // rdi
  int v19; // edx

  v5 = a2;
  v6 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v6 )
  {
    result = *(_QWORD *)(a1 + 16);
    v8 = 88LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v8 = 352;
  }
  for ( i = result + v8; i != result; result += 88LL )
  {
    if ( result )
      *(_QWORD *)result = -4;
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD *)v5;
      if ( *(_QWORD *)v5 != -4 && result != -16 )
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
        v14 = (unsigned int)v11 & ((unsigned int)result ^ (unsigned int)(result >> 9));
        v15 = v10 + 88 * v14;
        v16 = *(_QWORD *)v15;
        if ( result != *(_QWORD *)v15 )
        {
          while ( v16 != -4 )
          {
            if ( v16 == -16 && !v13 )
              v13 = v15;
            v14 = (unsigned int)v11 & (v12 + (_DWORD)v14);
            v15 = v10 + 88LL * (unsigned int)v14;
            v16 = *(_QWORD *)v15;
            if ( result == *(_QWORD *)v15 )
              goto LABEL_11;
            ++v12;
          }
          if ( v13 )
            v15 = v13;
        }
LABEL_11:
        *(_QWORD *)v15 = *(_QWORD *)v5;
        *(_QWORD *)(v15 + 8) = *(_QWORD *)(v5 + 8);
        *(_QWORD *)(v15 + 16) = *(_QWORD *)(v5 + 16);
        *(_QWORD *)(v15 + 24) = *(_QWORD *)(v5 + 24);
        v17 = *(_BYTE *)(v5 + 32);
        *(_QWORD *)(v15 + 48) = 0x400000000LL;
        *(_BYTE *)(v15 + 32) = v17;
        *(_QWORD *)(v15 + 40) = v15 + 56;
        if ( *(_DWORD *)(v5 + 48) )
        {
          v11 = v5 + 40;
          sub_D91460(v15 + 40, (char **)(v5 + 40), v14, v16, v10, v13);
        }
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
        v18 = *(_QWORD *)(v5 + 40);
        result = v5 + 56;
        if ( v18 != v5 + 56 )
          result = _libc_free(v18, v11);
      }
      v5 += 88;
    }
    while ( a3 != v5 );
  }
  return result;
}
