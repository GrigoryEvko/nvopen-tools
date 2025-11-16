// Function: sub_2E21F40
// Address: 0x2e21f40
//
__int64 __fastcall sub_2E21F40(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v6; // rax
  unsigned int v7; // eax
  unsigned int v8; // ecx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 i; // r10
  unsigned int v12; // eax
  __int16 v13; // cx
  unsigned int v14; // ecx
  __int64 v15; // rsi

  v2 = *(_QWORD *)(a2 + 32);
  result = 5LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  v4 = v2 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v4 != v2 )
  {
    do
    {
LABEL_6:
      if ( *(_BYTE *)v2 )
      {
        if ( *(_BYTE *)v2 == 12 )
          sub_2E21E10(a1, *(_QWORD *)(v2 + 24));
      }
      else if ( (*(_BYTE *)(v2 + 3) & 0x10) != 0 )
      {
        v6 = *(unsigned int *)(v2 + 8);
        if ( (unsigned int)(v6 - 1) <= 0x3FFFFFFE )
        {
          v7 = *(_DWORD *)(*(_QWORD *)(*a1 + 8) + 24 * v6 + 16);
          v8 = v7 & 0xFFF;
          v9 = *(_QWORD *)(*a1 + 56) + 2LL * (v7 >> 12);
          while ( v9 )
          {
            v9 += 2;
            *(_QWORD *)(a1[1] + 8LL * (v8 >> 6)) &= ~(1LL << v8);
            v8 += *(__int16 *)(v9 - 2);
            if ( !*(_WORD *)(v9 - 2) )
            {
              v2 += 40;
              if ( v4 != v2 )
                goto LABEL_6;
              goto LABEL_13;
            }
          }
        }
      }
      v2 += 40;
    }
    while ( v4 != v2 );
LABEL_13:
    v10 = *(_QWORD *)(a2 + 32);
    result = 5LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
    for ( i = v10 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF); i != v10; v10 += 40 )
    {
LABEL_14:
      if ( !*(_BYTE *)v10 )
      {
        result = *(unsigned __int8 *)(v10 + 4);
        if ( (result & 1) == 0
          && (result & 2) == 0
          && ((*(_BYTE *)(v10 + 3) & 0x10) == 0 || (*(_DWORD *)v10 & 0xFFF00) != 0) )
        {
          result = *(unsigned int *)(v10 + 8);
          if ( (unsigned int)(result - 1) <= 0x3FFFFFFE )
          {
            v12 = *(_DWORD *)(*(_QWORD *)(*a1 + 8) + 24 * result + 16);
            v13 = v12;
            result = v12 >> 12;
            v14 = v13 & 0xFFF;
            v15 = *(_QWORD *)(*a1 + 56) + 2 * result;
            while ( v15 )
            {
              v15 += 2;
              result = v14 >> 6;
              *(_QWORD *)(a1[1] + 8 * result) |= 1LL << v14;
              v14 += *(__int16 *)(v15 - 2);
              if ( !*(_WORD *)(v15 - 2) )
              {
                v10 += 40;
                if ( i != v10 )
                  goto LABEL_14;
                return result;
              }
            }
          }
        }
      }
    }
  }
  return result;
}
