// Function: sub_2043540
// Address: 0x2043540
//
__int64 __fastcall sub_2043540(__int64 *a1, __int64 *a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // rax
  unsigned int v5; // r8d
  __int64 v7; // r9
  __int64 v8; // r10
  int v9; // edi
  int v10; // esi
  int v11; // esi
  int v12; // esi
  __int64 v13; // rdi
  int v14; // r8d
  int v15; // r9d
  bool v16; // r10

  if ( *a1 && *a2 )
  {
    v4 = a2[4] - a1[4];
    *a4 = v4;
    if ( a1[2] != a2[2] || *((_DWORD *)a1 + 6) != *((_DWORD *)a2 + 6) || *((_BYTE *)a2 + 40) != *((_BYTE *)a1 + 40) )
      return 0;
    v7 = *a2;
    v8 = *a1;
    if ( *a1 == *a2 )
    {
      v5 = 1;
      if ( *((_DWORD *)a1 + 2) == *((_DWORD *)a2 + 2) )
        return v5;
    }
    v9 = *(unsigned __int16 *)(v8 + 24);
    v5 = v9 - 34;
    if ( (unsigned __int16)(v9 - 34) <= 1u || (unsigned __int16)(*(_WORD *)(v8 + 24) - 12) <= 1u )
    {
      v10 = *(unsigned __int16 *)(v7 + 24);
      v5 = v10 - 34;
      LOBYTE(v5) = (unsigned __int16)(v10 - 12) <= 1u || (unsigned __int16)(v10 - 34) <= 1u;
      if ( (_BYTE)v5 )
      {
        if ( *(_QWORD *)(v8 + 88) == *(_QWORD *)(v7 + 88) )
        {
          *a4 = *(_QWORD *)(v7 + 96) + v4 - *(_QWORD *)(v8 + 96);
          return v5;
        }
      }
    }
    if ( v9 == 38 || v9 == 16 )
    {
      LOBYTE(v5) = *(_WORD *)(v7 + 24) == 16 || *(_WORD *)(v7 + 24) == 38;
      if ( (_BYTE)v5 )
      {
        if ( *(int *)(v8 + 96) < 0 == *(int *)(v7 + 96) < 0 && *(_QWORD *)(v7 + 88) == *(_QWORD *)(v8 + 88) )
        {
          *a4 = (*(_DWORD *)(v7 + 96) & 0x7FFFFFFF) - (*(_DWORD *)(v8 + 96) & 0x7FFFFFFF) + v4;
          return v5;
        }
      }
    }
    if ( v9 != 36 && v9 != 14 )
      return 0;
    v11 = *(unsigned __int16 *)(v7 + 24);
    if ( v11 != 14 && v11 != 36 )
      return 0;
    v12 = *(_DWORD *)(v8 + 84);
    if ( v12 < 0
      && (v13 = *(_QWORD *)(*(_QWORD *)(a3 + 32) + 56LL), v14 = -*(_DWORD *)(v13 + 32), v14 <= v12)
      && (v15 = *(_DWORD *)(v7 + 84),
          v16 = v14 <= v15,
          v5 = (unsigned int)v15 >> 31,
          LOBYTE(v5) = v16 && v15 < 0,
          (_BYTE)v5) )
    {
      *a4 = *(_QWORD *)(*(_QWORD *)(v13 + 8) + 40LL * (unsigned int)(v15 + *(_DWORD *)(v13 + 32)))
          + v4
          - *(_QWORD *)(*(_QWORD *)(v13 + 8) + 40LL * (unsigned int)(v12 + *(_DWORD *)(v13 + 32)));
    }
    else
    {
      return 0;
    }
    return v5;
  }
  return 0;
}
