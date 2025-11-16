// Function: sub_3364290
// Address: 0x3364290
//
char __fastcall sub_3364290(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  char result; // al
  __int64 v5; // r8
  _DWORD *v6; // r10
  _DWORD *v7; // r9
  int v8; // eax
  int v9; // esi
  int v10; // edi
  __int64 v11; // r9

  if ( !*(_QWORD *)a1 )
    return 0;
  if ( !*(_QWORD *)a2 )
    return 0;
  if ( !*(_BYTE *)(a1 + 40) )
    return 0;
  result = *(_BYTE *)(a2 + 40);
  if ( !result )
    return 0;
  v5 = *(_QWORD *)(a2 + 32) - *(_QWORD *)(a1 + 32);
  *a4 = v5;
  if ( *(_QWORD *)(a2 + 16) != *(_QWORD *)(a1 + 16)
    || *(_DWORD *)(a2 + 24) != *(_DWORD *)(a1 + 24)
    || *(_BYTE *)(a2 + 48) != *(_BYTE *)(a1 + 48) )
  {
    return 0;
  }
  v6 = *(_DWORD **)a2;
  v7 = *(_DWORD **)a1;
  if ( *(_QWORD *)a2 != *(_QWORD *)a1 || *(_DWORD *)(a2 + 8) != *(_DWORD *)(a1 + 8) )
  {
    v8 = v7[6];
    if ( (unsigned int)(v8 - 13) > 1 && (unsigned int)(v8 - 37) > 1 )
    {
      if ( v8 == 41 || v8 == 17 )
      {
        result = v6[6] == 17 || v6[6] == 41;
        if ( result && (int)v7[26] < 0 == (int)v6[26] < 0 && *((_QWORD *)v7 + 12) == *((_QWORD *)v6 + 12) )
        {
          *a4 = (v6[26] & 0x7FFFFFFF) - (v7[26] & 0x7FFFFFFF) + v5;
          return result;
        }
      }
      else if ( v8 == 15 || v8 == 39 )
      {
        result = v6[6] == 39 || v6[6] == 15;
        if ( result )
        {
          v9 = v7[24];
          v10 = v6[24];
          if ( v9 == v10 )
            return result;
          if ( v9 < 0 )
          {
            v11 = *(_QWORD *)(*(_QWORD *)(a3 + 40) + 48LL);
            result = v9 >= -*(_DWORD *)(v11 + 32) && v10 < 0 && v10 >= -*(_DWORD *)(v11 + 32);
            if ( result )
            {
              *a4 = *(_QWORD *)(*(_QWORD *)(v11 + 8) + 40LL * (unsigned int)(*(_DWORD *)(v11 + 32) + v10))
                  + v5
                  - *(_QWORD *)(*(_QWORD *)(v11 + 8) + 40LL * (unsigned int)(v9 + *(_DWORD *)(v11 + 32)));
              return result;
            }
          }
        }
      }
    }
    else
    {
      result = (unsigned int)(v6[6] - 37) <= 1 || (unsigned int)(v6[6] - 13) <= 1;
      if ( result && *((_QWORD *)v7 + 12) == *((_QWORD *)v6 + 12) )
      {
        *a4 = *((_QWORD *)v6 + 13) + v5 - *((_QWORD *)v7 + 13);
        return result;
      }
    }
    return 0;
  }
  return result;
}
