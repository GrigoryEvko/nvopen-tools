// Function: sub_D60340
// Address: 0xd60340
//
__int64 __fastcall sub_D60340(__int64 a1, unsigned int a2)
{
  __int64 v4; // r12
  char v5; // si
  unsigned __int64 v6; // rdx
  unsigned int v7; // r14d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v12; // rax
  __int64 v13; // rcx
  _QWORD *v14; // r12
  __int64 v15; // rdx
  __int64 v16; // rax
  _BYTE v17[352]; // [rsp+0h] [rbp-160h] BYREF

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v12 = a1 + 16;
    v13 = a1 + 336;
  }
  else
  {
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
    a2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v12 = a1 + 16;
      v13 = a1 + 336;
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 40LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        a2 = 64;
        v8 = 2560;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = a2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 40LL * v7;
        sub_D60170(a1, v4, v4 + v10);
        return sub_C7D6A0(v4, v10, 8);
      }
      v12 = a1 + 16;
      v13 = a1 + 336;
      a2 = 64;
    }
  }
  v14 = v17;
  do
  {
    v15 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 != -4096 && v15 != -8192 )
    {
      if ( v14 )
        *v14 = v15;
      v14 += 5;
      *((_DWORD *)v14 - 6) = *(_DWORD *)(v12 + 16);
      *(v14 - 4) = *(_QWORD *)(v12 + 8);
      *((_DWORD *)v14 - 2) = *(_DWORD *)(v12 + 32);
      *(v14 - 2) = *(_QWORD *)(v12 + 24);
    }
    v12 += 40;
  }
  while ( v12 != v13 );
  if ( a2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v16 = sub_C7D670(40LL * a2, 8);
    *(_DWORD *)(a1 + 24) = a2;
    *(_QWORD *)(a1 + 16) = v16;
  }
  return sub_D60170(a1, (__int64)v17, (__int64)v14);
}
