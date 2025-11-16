// Function: sub_A3F3B0
// Address: 0xa3f3b0
//
__int64 __fastcall sub_A3F3B0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rsi
  __int64 v4; // r8
  int v5; // ecx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // rdi
  __int64 v10; // rcx
  unsigned int v11; // edi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  int v15; // eax
  int v16; // r9d
  int v17; // eax
  int v18; // r10d

  if ( *(_BYTE *)a2 == 24 )
  {
    v2 = *(_DWORD *)(a1 + 280);
    v3 = *(_QWORD *)(a2 + 24);
    v4 = *(_QWORD *)(a1 + 264);
    if ( v2 )
    {
      v5 = v2 - 1;
      v6 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( v3 == *v7 )
        return (unsigned int)(*((_DWORD *)v7 + 3) - 1);
      v15 = 1;
      while ( v8 != -4096 )
      {
        v16 = v15 + 1;
        v6 = v5 & (v15 + v6);
        v7 = (__int64 *)(v4 + 16LL * v6);
        v8 = *v7;
        if ( v3 == *v7 )
          return (unsigned int)(*((_DWORD *)v7 + 3) - 1);
        v15 = v16;
      }
    }
    return 0xFFFFFFFFLL;
  }
  else
  {
    v10 = *(_QWORD *)(a1 + 88);
    v11 = *(_DWORD *)(a1 + 104);
    if ( v11 )
    {
      v12 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( a2 == *v13 )
        return (unsigned int)(*((_DWORD *)v13 + 2) - 1);
      v17 = 1;
      while ( v14 != -4096 )
      {
        v18 = v17 + 1;
        v12 = (v11 - 1) & (v17 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( a2 == *v13 )
          return (unsigned int)(*((_DWORD *)v13 + 2) - 1);
        v17 = v18;
      }
    }
    return (unsigned int)(*(_DWORD *)(v10 + 16LL * v11 + 8) - 1);
  }
}
