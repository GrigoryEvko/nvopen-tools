// Function: sub_1A1C610
// Address: 0x1a1c610
//
__int64 __fastcall sub_1A1C610(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r14
  unsigned __int64 v6; // rax
  int v7; // ebx
  __int64 v8; // rdi
  __int64 *v9; // rax
  __int64 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // r13d
  __int64 v14; // rax
  __int64 v15[44]; // [rsp+0h] [rbp-160h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v13 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
LABEL_16:
    sub_1A1B2B0(a1, v5, &v5[5 * v13]);
    return j___libc_free_0(v5);
  }
  v5 = *(__int64 **)(a1 + 16);
  v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
  v7 = v6;
  if ( (unsigned int)v6 <= 0x40 )
  {
    if ( v4 )
    {
      v8 = 2560;
      v7 = 64;
      goto LABEL_5;
    }
    v13 = *(_DWORD *)(a1 + 24);
    v7 = 64;
    v8 = 2560;
    goto LABEL_20;
  }
  v8 = 40LL * (unsigned int)v6;
  if ( !v4 )
  {
    v13 = *(_DWORD *)(a1 + 24);
LABEL_20:
    v14 = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v14;
    goto LABEL_16;
  }
LABEL_5:
  v9 = (__int64 *)(a1 + 16);
  v10 = v15;
  do
  {
    v11 = *v9;
    if ( *v9 != -8 && v11 != -16 )
    {
      if ( v10 )
        *v10 = v11;
      v10 += 5;
      *(v10 - 4) = v9[1];
      *(v10 - 3) = v9[2];
      *(v10 - 2) = v9[3];
      *(v10 - 1) = v9[4];
    }
    v9 += 5;
  }
  while ( v9 != (__int64 *)(a1 + 336) );
  *(_BYTE *)(a1 + 8) &= ~1u;
  v12 = sub_22077B0(v8);
  *(_DWORD *)(a1 + 24) = v7;
  *(_QWORD *)(a1 + 16) = v12;
  return sub_1A1B2B0(a1, v15, v10);
}
