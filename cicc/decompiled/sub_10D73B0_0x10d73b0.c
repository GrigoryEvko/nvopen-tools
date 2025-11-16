// Function: sub_10D73B0
// Address: 0x10d73b0
//
bool __fastcall sub_10D73B0(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  unsigned __int8 *v4; // rsi
  unsigned __int8 *v5; // rdx
  int v6; // ecx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9

  result = 0;
  if ( a2 + 29 == *a3 )
  {
    if ( (v4 = (unsigned __int8 *)*((_QWORD *)a3 - 8),
          v5 = (unsigned __int8 *)*((_QWORD *)a3 - 4),
          v6 = *(_DWORD *)(a1 + 16) + 29,
          *v4 != v6)
      || ((v9 = *((_QWORD *)v4 - 8), v10 = *((_QWORD *)v4 - 4), v9 != *(_QWORD *)a1) || *(_QWORD *)(a1 + 8) != v10)
      && (*(_QWORD *)a1 != v10 || v9 != *(_QWORD *)(a1 + 8))
      || (result = 1, *(unsigned __int8 **)(a1 + 24) != v5) )
    {
      result = 0;
      if ( v6 == *v5 )
      {
        v7 = *((_QWORD *)v5 - 8);
        v8 = *((_QWORD *)v5 - 4);
        if ( v7 == *(_QWORD *)a1 && v8 == *(_QWORD *)(a1 + 8) )
          return *(_QWORD *)(a1 + 24) == (_QWORD)v4;
        result = 0;
        if ( v8 == *(_QWORD *)a1 && v7 == *(_QWORD *)(a1 + 8) )
          return *(_QWORD *)(a1 + 24) == (_QWORD)v4;
      }
    }
  }
  return result;
}
