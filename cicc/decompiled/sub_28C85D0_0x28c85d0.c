// Function: sub_28C85D0
// Address: 0x28c85d0
//
bool __fastcall sub_28C85D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  int v5; // ecx
  int v6; // ecx
  unsigned int v7; // r8d
  __int64 *v8; // rax
  __int64 v9; // r9
  unsigned int v10; // r9d
  unsigned int v11; // esi
  __int64 *v12; // rax
  __int64 v13; // r8
  int v15; // eax
  int v16; // eax
  int v17; // r10d
  int v18; // r10d

  v4 = *(_QWORD *)(a1 + 2360);
  v5 = *(_DWORD *)(a1 + 2376);
  if ( v5 )
  {
    v6 = v5 - 1;
    v7 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      v10 = *((_DWORD *)v8 + 2);
    }
    else
    {
      v16 = 1;
      while ( v9 != -4096 )
      {
        v18 = v16 + 1;
        v7 = v6 & (v16 + v7);
        v8 = (__int64 *)(v4 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v16 = v18;
      }
      v10 = 0;
    }
    v11 = v6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v12 = (__int64 *)(v4 + 16LL * v11);
    v13 = *v12;
    if ( a3 == *v12 )
      return v10 < *((_DWORD *)v12 + 2);
    v15 = 1;
    while ( v13 != -4096 )
    {
      v17 = v15 + 1;
      v11 = v6 & (v15 + v11);
      v12 = (__int64 *)(v4 + 16LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        return v10 < *((_DWORD *)v12 + 2);
      v15 = v17;
    }
  }
  return 0;
}
