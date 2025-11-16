// Function: sub_2ACF7C0
// Address: 0x2acf7c0
//
__int64 __fastcall sub_2ACF7C0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 i; // rdx
  __int64 j; // rbx
  int v12; // edx
  int v13; // eax
  char v14; // r8
  int v15; // esi
  int v16; // ecx
  __int64 v17; // rdi
  int v18; // r10d
  int *v19; // r9
  unsigned int k; // esi
  int *v21; // rax
  int v22; // r11d
  char v23; // dl
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 m; // rdx
  unsigned int v28; // esi

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = sub_AF1560((unsigned int)(a2 - 1));
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = sub_C7D670(40LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = 40 * v3;
    v9 = v4 + 40 * v3;
    for ( i = result + 40 * v7; i != result; result += 40 )
    {
      if ( result )
      {
        *(_DWORD *)result = -1;
        *(_BYTE *)(result + 4) = 1;
      }
    }
    if ( v9 != v4 )
    {
      for ( j = v4; v9 != j; j += 40 )
      {
        v12 = *(_DWORD *)j;
        if ( *(_DWORD *)j == -1 )
        {
          if ( !*(_BYTE *)(j + 4) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
              goto LABEL_43;
            v14 = 0;
            v15 = -37;
            goto LABEL_14;
          }
        }
        else
        {
          if ( v12 != -2 )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
              goto LABEL_43;
            v14 = *(_BYTE *)(j + 4);
            v15 = 37 * v12;
            if ( v14 )
              goto LABEL_32;
            goto LABEL_14;
          }
          if ( *(_BYTE *)(j + 4) )
          {
            v13 = *(_DWORD *)(a1 + 24);
            if ( !v13 )
            {
LABEL_43:
              MEMORY[0] = *(_DWORD *)j;
              MEMORY[4] = *(_BYTE *)(j + 4);
              BUG();
            }
            v14 = *(_BYTE *)(j + 4);
            v15 = -74;
LABEL_32:
            --v15;
LABEL_14:
            v16 = v13 - 1;
            v18 = 1;
            v19 = 0;
            for ( k = (v13 - 1) & v15; ; k = v16 & v28 )
            {
              v17 = *(_QWORD *)(a1 + 8);
              v21 = (int *)(v17 + 40LL * k);
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
                v19 = (int *)(v17 + 40LL * k);
              }
              v28 = v18 + k;
              ++v18;
            }
            *v21 = *(_DWORD *)j;
            v23 = *(_BYTE *)(j + 4);
            *((_QWORD *)v21 + 3) = 0;
            *((_QWORD *)v21 + 2) = 0;
            v21[8] = 0;
            *((_BYTE *)v21 + 4) = v23;
            *((_QWORD *)v21 + 1) = 1;
            v24 = *(_QWORD *)(j + 16);
            ++*(_QWORD *)(j + 8);
            v25 = *((_QWORD *)v21 + 2);
            *((_QWORD *)v21 + 2) = v24;
            LODWORD(v24) = *(_DWORD *)(j + 24);
            *(_QWORD *)(j + 16) = v25;
            LODWORD(v25) = v21[6];
            v21[6] = v24;
            LODWORD(v24) = *(_DWORD *)(j + 28);
            *(_DWORD *)(j + 24) = v25;
            LODWORD(v25) = v21[7];
            v21[7] = v24;
            LODWORD(v24) = *(_DWORD *)(j + 32);
            *(_DWORD *)(j + 28) = v25;
            LODWORD(v25) = v21[8];
            v21[8] = v24;
            *(_DWORD *)(j + 32) = v25;
            ++*(_DWORD *)(a1 + 16);
            sub_C7D6A0(*(_QWORD *)(j + 16), 24LL * *(unsigned int *)(j + 32), 8);
          }
        }
      }
    }
    return sub_C7D6A0(v4, v8, 8);
  }
  else
  {
    v26 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = result + 40 * v26; m != result; result += 40 )
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
