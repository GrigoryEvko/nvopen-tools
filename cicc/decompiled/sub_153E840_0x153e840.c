// Function: sub_153E840
// Address: 0x153e840
//
__int64 __fastcall sub_153E840(__int64 a1, __int64 a2)
{
  int v2; // edx
  __int64 result; // rax
  __int64 v4; // rsi
  int v5; // ecx
  __int64 v6; // r8
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r8
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdi
  int v15; // eax
  int v16; // eax
  int v17; // r10d
  int v18; // r9d

  if ( *(_BYTE *)(a2 + 16) != 19 )
  {
    v10 = *(unsigned int *)(a1 + 104);
    v11 = *(_QWORD *)(a1 + 88);
    if ( (_DWORD)v10 )
    {
      v12 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
        return (unsigned int)(*((_DWORD *)v13 + 2) - 1);
      v16 = 1;
      while ( v14 != -8 )
      {
        v17 = v16 + 1;
        v12 = (v10 - 1) & (v16 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          return (unsigned int)(*((_DWORD *)v13 + 2) - 1);
        v16 = v17;
      }
    }
    v13 = (__int64 *)(v11 + 16 * v10);
    return (unsigned int)(*((_DWORD *)v13 + 2) - 1);
  }
  v2 = *(_DWORD *)(a1 + 280);
  result = 0xFFFFFFFFLL;
  if ( v2 )
  {
    v4 = *(_QWORD *)(a2 + 24);
    v5 = v2 - 1;
    v6 = *(_QWORD *)(a1 + 264);
    v7 = (v2 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
    {
      return (unsigned int)(*((_DWORD *)v8 + 3) - 1);
    }
    else
    {
      v15 = 1;
      while ( v9 != -4 )
      {
        v18 = v15 + 1;
        v7 = v5 & (v15 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v4 == *v8 )
          return (unsigned int)(*((_DWORD *)v8 + 3) - 1);
        v15 = v18;
      }
      return 0xFFFFFFFFLL;
    }
  }
  return result;
}
