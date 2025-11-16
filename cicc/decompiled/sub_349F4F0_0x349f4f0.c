// Function: sub_349F4F0
// Address: 0x349f4f0
//
void __fastcall sub_349F4F0(__int64 a1, unsigned int a2)
{
  unsigned int v3; // ebx
  char v4; // cl
  unsigned __int64 v5; // rax
  int *v6; // r14
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // r13
  int *v13; // r13
  __int64 v14; // rax
  int v15[40]; // [rsp+0h] [rbp-A0h] BYREF

  v3 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v10 = a1 + 16;
    v11 = a1 + 144;
    if ( !v4 )
    {
      v6 = *(int **)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
  }
  else
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v3 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v10 = a1 + 16;
      v11 = a1 + 144;
      if ( !v4 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 32LL * (unsigned int)v5;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v4 )
      {
        v6 = *(int **)(a1 + 16);
        v7 = *(unsigned int *)(a1 + 24);
        v3 = 64;
        v8 = 2048;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v3;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v12 = 8 * v7;
        sub_349EBA0(a1, v6, &v6[v12]);
        sub_C7D6A0((__int64)v6, v12 * 4, 8);
        return;
      }
      v10 = a1 + 16;
      v11 = a1 + 144;
      v3 = 64;
    }
  }
  v13 = v15;
  do
  {
    if ( *(_DWORD *)v10 <= 0xFFFFFFFD )
    {
      if ( v13 )
        *v13 = *(_DWORD *)v10;
      v13 += 8;
      *((_QWORD *)v13 - 3) = *(_QWORD *)(v10 + 8);
      *((_QWORD *)v13 - 2) = *(_QWORD *)(v10 + 16);
      *((_QWORD *)v13 - 1) = *(_QWORD *)(v10 + 24);
    }
    v10 += 32;
  }
  while ( v11 != v10 );
  if ( v3 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v14 = sub_C7D670(32LL * v3, 8);
    *(_DWORD *)(a1 + 24) = v3;
    *(_QWORD *)(a1 + 16) = v14;
  }
  sub_349EBA0(a1, v15, v13);
}
