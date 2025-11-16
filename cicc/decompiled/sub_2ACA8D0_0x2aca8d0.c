// Function: sub_2ACA8D0
// Address: 0x2aca8d0
//
__int64 __fastcall sub_2ACA8D0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 i; // rdx
  __int64 j; // rbx
  int v12; // eax
  int v13; // edx
  char v14; // r8
  int v15; // ecx
  int v16; // edx
  __int64 v17; // rdi
  int v18; // r10d
  int *v19; // r9
  unsigned int k; // ecx
  int *v21; // rsi
  int v22; // r11d
  __int64 v23; // rdi
  _DWORD *v24; // rsi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 m; // rdx
  unsigned int v28; // ecx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(72LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 72 * v3;
    v9 = v4 + 72 * v3;
    for ( i = result + 72 * v7; i != result; result += 72 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_BYTE *)(result + 4) = 1;
      }
    }
    if ( v9 != v4 )
    {
      for ( j = v4; v9 != j; j += 72 )
      {
        while ( 1 )
        {
          v12 = *(_DWORD *)j;
          if ( *(_DWORD *)j != -1 )
            break;
          if ( !*(_BYTE *)(j + 4) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
              goto LABEL_45;
            v14 = 0;
            v15 = -37;
            goto LABEL_14;
          }
LABEL_27:
          j += 72;
          if ( v9 == j )
            return sub_C7D6A0(v4, v8, 8);
        }
        if ( v12 == -2 )
        {
          if ( *(_BYTE *)(j + 4) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
LABEL_45:
              MEMORY[0] = *(_DWORD *)j;
              MEMORY[4] = *(_BYTE *)(j + 4);
              BUG();
            }
            v14 = *(_BYTE *)(j + 4);
            v15 = -74;
LABEL_32:
            --v15;
            goto LABEL_14;
          }
          goto LABEL_27;
        }
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
          goto LABEL_45;
        v14 = *(_BYTE *)(j + 4);
        v15 = 37 * v12;
        if ( v14 )
          goto LABEL_32;
LABEL_14:
        v16 = v13 - 1;
        v18 = 1;
        v19 = 0;
        for ( k = v16 & v15; ; k = v16 & v28 )
        {
          v17 = *(_QWORD *)(a1 + 8);
          v21 = (int *)(v17 + 72LL * k);
          v22 = *v21;
          if ( v12 == *v21 && v14 == *((_BYTE *)v21 + 4) )
            break;
          if ( v22 == -1 )
          {
            if ( *((_BYTE *)v21 + 4) )
            {
              if ( v19 )
                v21 = v19;
              break;
            }
          }
          else if ( v22 == -2 && *((_BYTE *)v21 + 4) != 1 && !v19 )
          {
            v19 = (int *)(v17 + 72LL * k);
          }
          v28 = v18 + k;
          ++v18;
        }
        v23 = (__int64)(v21 + 2);
        v24 = v21 + 10;
        *(v24 - 10) = *(_DWORD *)j;
        *((_BYTE *)v24 - 36) = *(_BYTE *)(j + 4);
        sub_C8CF70(v23, v24, 4, j + 40, j + 8);
        ++*(_DWORD *)(a1 + 16);
        if ( *(_BYTE *)(j + 36) )
          goto LABEL_27;
        v25 = *(_QWORD *)(j + 16);
        _libc_free(v25);
      }
    }
    return sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + 72 * v26; m != result; result += 72 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_BYTE *)(result + 4) = 1;
      }
    }
  }
  return result;
}
