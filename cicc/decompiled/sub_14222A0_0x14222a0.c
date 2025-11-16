// Function: sub_14222A0
// Address: 0x14222a0
//
__int64 *__fastcall sub_14222A0(__int64 a1, __int64 a2)
{
  int v3; // eax
  int v5; // ecx
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  int v10; // edx
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // esi
  __int64 *result; // rax
  __int64 v18; // rdx
  int v19; // eax
  int v20; // eax
  int v21; // r8d
  int v22; // r10d

  v3 = *(_DWORD *)(a1 + 320);
  if ( v3 )
  {
    v5 = v3 - 1;
    v6 = *(_QWORD *)(a1 + 304);
    v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( *v8 == a2 )
    {
LABEL_3:
      *v8 = -16;
      --*(_DWORD *)(a1 + 312);
      ++*(_DWORD *)(a1 + 316);
    }
    else
    {
      v20 = 1;
      while ( v9 != -8 )
      {
        v21 = v20 + 1;
        v7 = v5 & (v20 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  v10 = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned int)(v10 - 21) <= 1 )
  {
    if ( *(_QWORD *)(a2 - 24) )
    {
      v11 = *(_QWORD *)(a2 - 16);
      v12 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v12 = v11;
      if ( v11 )
        *(_QWORD *)(v11 + 16) = *(_QWORD *)(v11 + 16) & 3LL | v12;
      LOBYTE(v10) = *(_BYTE *)(a2 + 16);
    }
    *(_QWORD *)(a2 - 24) = 0;
  }
  if ( (_BYTE)v10 == 21
    || (sub_141FE00(*(_QWORD *)(a1 + 328), a2), (unsigned int)*(unsigned __int8 *)(a2 + 16) - 21 <= 1) )
  {
    v13 = *(_QWORD *)(a2 + 72);
  }
  else
  {
    v13 = *(_QWORD *)(a2 + 64);
  }
  v14 = *(unsigned int *)(a1 + 48);
  v15 = *(_QWORD *)(a1 + 32);
  if ( (_DWORD)v14 )
  {
    v16 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    result = (__int64 *)(v15 + 16LL * v16);
    v18 = *result;
    if ( v13 == *result )
    {
LABEL_14:
      if ( result[1] != a2 )
        return result;
LABEL_19:
      *result = -16;
      --*(_DWORD *)(a1 + 40);
      ++*(_DWORD *)(a1 + 44);
      return result;
    }
    v19 = 1;
    while ( v18 != -8 )
    {
      v22 = v19 + 1;
      v16 = (v14 - 1) & (v19 + v16);
      result = (__int64 *)(v15 + 16LL * v16);
      v18 = *result;
      if ( v13 == *result )
        goto LABEL_14;
      v19 = v22;
    }
  }
  result = (__int64 *)(v15 + 16 * v14);
  if ( result[1] == a2 )
    goto LABEL_19;
  return result;
}
