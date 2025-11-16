// Function: sub_2F91510
// Address: 0x2f91510
//
__int64 __fastcall sub_2F91510(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 i; // rbx
  unsigned int v7; // r12d
  __int64 v8; // r13
  unsigned int v9; // ecx
  __int16 *v10; // rsi
  int v11; // edx
  unsigned int v12; // ecx
  __int64 v13; // rdx

  v3 = *(_QWORD *)(a3 + 32);
  result = 5LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF);
  for ( i = v3 + 40LL * (*(_DWORD *)(a3 + 40) & 0xFFFFFF); i != v3; v3 += 40 )
  {
    if ( !*(_BYTE *)v3 )
    {
      result = *(unsigned __int8 *)(v3 + 4);
      if ( (*(_BYTE *)(v3 + 4) & 1) == 0
        && (result & 2) == 0
        && ((*(_BYTE *)(v3 + 3) & 0x10) == 0 || (*(_DWORD *)v3 & 0xFFF00) != 0) )
      {
        v7 = *(_DWORD *)(v3 + 8);
        if ( v7 )
        {
          v8 = 24LL * v7;
          v9 = *(_DWORD *)(*(_QWORD *)(*a2 + 8LL) + v8 + 16) & 0xFFF;
          v10 = (__int16 *)(*(_QWORD *)(*a2 + 56LL) + 2LL * (*(_DWORD *)(*(_QWORD *)(*a2 + 8LL) + v8 + 16) >> 12));
          do
          {
            if ( !v10 )
              break;
            if ( (*(_QWORD *)(a2[1] + 8LL * (v9 >> 6)) & (1LL << v9)) != 0 )
              break;
            v11 = *v10++;
            v9 += v11;
          }
          while ( (_WORD)v11 );
          *(_BYTE *)(v3 + 3) = *(_BYTE *)(v3 + 3) & 0xBF | ((*(_BYTE *)(v3 + 4) & 1) << 6);
          result = *(_DWORD *)(*(_QWORD *)(*a2 + 8LL) + v8 + 16) >> 12;
          v12 = *(_DWORD *)(*(_QWORD *)(*a2 + 8LL) + v8 + 16) & 0xFFF;
          v13 = *(_QWORD *)(*a2 + 56LL) + 2 * result;
          do
          {
            if ( !v13 )
              break;
            v13 += 2;
            result = v12 >> 6;
            *(_QWORD *)(a2[1] + 8 * result) |= 1LL << v12;
            v12 += *(__int16 *)(v13 - 2);
          }
          while ( *(_WORD *)(v13 - 2) );
        }
      }
    }
  }
  return result;
}
