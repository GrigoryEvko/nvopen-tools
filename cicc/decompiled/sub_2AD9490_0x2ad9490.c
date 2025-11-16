// Function: sub_2AD9490
// Address: 0x2ad9490
//
__int64 __fastcall sub_2AD9490(__int64 a1, int a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r10
  __int64 i; // rdx
  __int64 j; // rax
  int v12; // edx
  int v13; // ecx
  char v14; // r9
  __int64 v15; // r11
  int v16; // esi
  int v17; // ecx
  int v18; // r15d
  int *v19; // r14
  unsigned int k; // esi
  int *v21; // rdi
  int v22; // r8d
  __int64 v23; // rdx
  __int64 m; // rdx
  unsigned int v25; // esi

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(8LL * v5, 4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    v8 = 8 * v3;
    *(_QWORD *)(a1 + 16) = 0;
    v9 = v4 + v8;
    for ( i = result + 8 * v7; i != result; result += 8 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_BYTE *)(result + 4) = 1;
      }
    }
    for ( j = v4; v9 != j; j += 8 )
    {
      v12 = *(_DWORD *)j;
      if ( *(_DWORD *)j == -1 )
      {
        v14 = *(_BYTE *)(j + 4);
        if ( !v14 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          v15 = *(_QWORD *)(a1 + 8);
          if ( !v13 )
            goto LABEL_42;
          v16 = -37;
          goto LABEL_13;
        }
      }
      else
      {
        if ( v12 != -2 )
        {
          v13 = *(_DWORD *)(a1 + 24);
          if ( !v13 )
            goto LABEL_42;
          v14 = *(_BYTE *)(j + 4);
          v15 = *(_QWORD *)(a1 + 8);
          v16 = 37 * v12;
          if ( v14 )
            goto LABEL_31;
          goto LABEL_13;
        }
        if ( *(_BYTE *)(j + 4) )
        {
          v13 = *(_DWORD *)(a1 + 24);
          v15 = *(_QWORD *)(a1 + 8);
          if ( !v13 )
          {
LABEL_42:
            MEMORY[0] = *(_DWORD *)j;
            MEMORY[4] = *(_BYTE *)(j + 4);
            BUG();
          }
          v16 = -74;
LABEL_31:
          --v16;
          v14 = 1;
LABEL_13:
          v17 = v13 - 1;
          v18 = 1;
          v19 = 0;
          for ( k = v17 & v16; ; k = v17 & v25 )
          {
            v21 = (int *)(v15 + 8LL * k);
            v22 = *v21;
            if ( v12 == *v21 && *((_BYTE *)v21 + 4) == v14 )
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
              v19 = (int *)(v15 + 8LL * k);
            }
            v25 = v18 + k;
            ++v18;
          }
          *v21 = *(_DWORD *)j;
          *((_BYTE *)v21 + 4) = *(_BYTE *)(j + 4);
          ++*(_DWORD *)(a1 + 16);
        }
      }
    }
    return sub_C7D6A0(v4, v8, 4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + 8 * v23; m != result; result += 8 )
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
