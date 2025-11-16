// Function: sub_3373C30
// Address: 0x3373c30
//
__int64 __fastcall sub_3373C30(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  int v6; // eax
  int v7; // ecx
  unsigned int v8; // r8d
  int v9; // edi
  unsigned int v10; // eax
  __int64 v11; // rdx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rsi
  int v16; // eax
  int v17; // ecx
  int v18; // edi
  unsigned int v19; // eax
  __int64 v20; // rdx

  if ( *(_BYTE *)a2 <= 0x1Cu )
  {
    if ( *(_BYTE *)a2 == 22 )
    {
      LOBYTE(v13) = sub_AA5B70(a3);
      v8 = v13;
      if ( !(_BYTE)v13 )
      {
        v14 = *(_QWORD *)(a1 + 960);
        v15 = *(_QWORD *)(v14 + 128);
        v16 = *(_DWORD *)(v14 + 144);
        if ( !v16 )
          return v8;
        v17 = v16 - 1;
        v18 = 1;
        v19 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v20 = *(_QWORD *)(v15 + 16LL * v19);
        if ( a2 != v20 )
        {
          while ( v20 != -4096 )
          {
            v19 = v17 & (v18 + v19);
            v20 = *(_QWORD *)(v15 + 16LL * v19);
            if ( a2 == v20 )
              return 1;
            ++v18;
          }
          return v8;
        }
      }
    }
    return 1;
  }
  if ( a3 == *(_QWORD *)(a2 + 40) )
    return 1;
  v4 = *(_QWORD *)(a1 + 960);
  v5 = *(_QWORD *)(v4 + 128);
  v6 = *(_DWORD *)(v4 + 144);
  if ( v6 )
  {
    v7 = v6 - 1;
    v8 = 1;
    v9 = 1;
    v10 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = *(_QWORD *)(v5 + 16LL * v10);
    if ( v11 == a2 )
      return v8;
    while ( v11 != -4096 )
    {
      v10 = v7 & (v9 + v10);
      v11 = *(_QWORD *)(v5 + 16LL * v10);
      if ( a2 == v11 )
        return 1;
      ++v9;
    }
  }
  return 0;
}
