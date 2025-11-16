// Function: sub_2F4F320
// Address: 0x2f4f320
//
__int64 **__fastcall sub_2F4F320(__int64 **a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 **v8; // rsi
  __int64 v9; // rax
  int v10; // edx
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 **result; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  bool v19; // zf

  v5 = a2 - (_QWORD)a1;
  v6 = (a2 - (__int64)a1) >> 5;
  v7 = v5 >> 3;
  if ( v6 <= 0 )
  {
LABEL_17:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return (__int64 **)a2;
LABEL_29:
        v18 = **a1;
        if ( v18 && (v18 & 4) != 0 )
        {
          v19 = *(_BYTE *)(*(_QWORD *)(a3 + 8)
                         + 40LL * (unsigned int)(*(_DWORD *)((v18 & 0xFFFFFFFFFFFFFFF8LL) + 16) + *(_DWORD *)(a3 + 32))
                         + 18) == 0;
          result = a1;
          if ( v19 )
            return (__int64 **)a2;
          return result;
        }
LABEL_38:
        BUG();
      }
      v16 = **a1;
      if ( !v16 || (v16 & 4) == 0 )
        goto LABEL_38;
      if ( *(_BYTE *)(*(_QWORD *)(a3 + 8)
                    + 40LL * (unsigned int)(*(_DWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 16) + *(_DWORD *)(a3 + 32))
                    + 18) )
        return a1;
      ++a1;
    }
    v17 = **a1;
    if ( !v17 || (v17 & 4) == 0 )
      goto LABEL_38;
    if ( !*(_BYTE *)(*(_QWORD *)(a3 + 8)
                   + 40LL * (unsigned int)(*(_DWORD *)((v17 & 0xFFFFFFFFFFFFFFF8LL) + 16) + *(_DWORD *)(a3 + 32))
                   + 18) )
    {
      ++a1;
      goto LABEL_29;
    }
    return a1;
  }
  v8 = &a1[4 * v6];
  while ( 1 )
  {
    v9 = **a1;
    if ( !v9 || (v9 & 4) == 0 )
      goto LABEL_38;
    v10 = *(_DWORD *)(a3 + 32);
    v11 = *(_QWORD *)(a3 + 8);
    if ( *(_BYTE *)(v11 + 40LL * (unsigned int)(*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 16) + v10) + 18) )
      return a1;
    v12 = *a1[1];
    if ( !v12 || (v12 & 4) == 0 )
      goto LABEL_38;
    if ( *(_BYTE *)(v11 + 40LL * (unsigned int)(*(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 16) + v10) + 18) )
      return a1 + 1;
    v13 = *a1[2];
    if ( !v13 || (v13 & 4) == 0 )
      goto LABEL_38;
    if ( *(_BYTE *)(v11 + 40LL * (unsigned int)(*(_DWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 16) + v10) + 18) )
      return a1 + 2;
    v14 = *a1[3];
    if ( !v14 || (v14 & 4) == 0 )
      goto LABEL_38;
    if ( *(_BYTE *)(v11 + 40LL * (unsigned int)(*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 16) + v10) + 18) )
      return a1 + 3;
    a1 += 4;
    if ( a1 == v8 )
    {
      v7 = (a2 - (__int64)a1) >> 3;
      goto LABEL_17;
    }
  }
}
