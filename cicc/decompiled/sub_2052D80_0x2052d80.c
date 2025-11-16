// Function: sub_2052D80
// Address: 0x2052d80
//
__int64 __fastcall sub_2052D80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v3; // cl
  __int64 result; // rax
  __int64 v5; // rcx
  int v6; // edx
  __int64 v7; // r8
  int v8; // ecx
  unsigned int v9; // edx
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rcx
  int v13; // edx
  __int64 v14; // r8
  int v15; // ecx
  unsigned int v16; // edx
  __int64 v17; // rdi
  int v18; // eax
  int v19; // eax

  v3 = *(_BYTE *)(a2 + 16);
  result = 1;
  if ( v3 > 0x17u )
  {
    if ( *(_QWORD *)(a2 + 40) == a3 )
      return result;
    v5 = *(_QWORD *)(a1 + 712);
    result = 0;
    v6 = *(_DWORD *)(v5 + 232);
    if ( !v6 )
      return result;
    v7 = *(_QWORD *)(v5 + 216);
    v8 = v6 - 1;
    v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = *(_QWORD *)(v7 + 16LL * v9);
    result = 1;
    if ( a2 == v10 )
      return result;
    v19 = 1;
    while ( v10 != -8 )
    {
      v9 = v8 & (v19 + v9);
      v10 = *(_QWORD *)(v7 + 16LL * v9);
      if ( a2 == v10 )
        return 1;
      ++v19;
    }
    return 0;
  }
  if ( v3 == 17 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a3 + 56) + 80LL);
    if ( !v11 || a3 != v11 - 24 )
    {
      v12 = *(_QWORD *)(a1 + 712);
      result = 0;
      v13 = *(_DWORD *)(v12 + 232);
      if ( v13 )
      {
        v14 = *(_QWORD *)(v12 + 216);
        v15 = v13 - 1;
        v16 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v17 = *(_QWORD *)(v14 + 16LL * v16);
        result = 1;
        if ( a2 != v17 )
        {
          v18 = 1;
          while ( v17 != -8 )
          {
            v16 = v15 & (v18 + v16);
            v17 = *(_QWORD *)(v14 + 16LL * v16);
            if ( a2 == v17 )
              return 1;
            ++v18;
          }
          return 0;
        }
      }
    }
  }
  return result;
}
