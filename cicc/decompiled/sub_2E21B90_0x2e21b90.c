// Function: sub_2E21B90
// Address: 0x2e21b90
//
__int64 __fastcall sub_2E21B90(_QWORD *a1, __int64 a2)
{
  unsigned int *v3; // r12
  __int64 result; // rax
  unsigned int *v5; // rdi
  __int64 v6; // r10
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // r8
  unsigned int v10; // ecx
  __int64 v11; // rsi
  int v12; // r13d

  v3 = *(unsigned int **)(a2 + 192);
  result = sub_2E33140(a2);
  if ( v3 != (unsigned int *)result )
  {
    v5 = (unsigned int *)result;
    do
    {
      v6 = *((_QWORD *)v5 + 1);
      v7 = *((_QWORD *)v5 + 2);
      v8 = *(_QWORD *)(*a1 + 8LL) + 24LL * *v5;
      v9 = *(_QWORD *)(*a1 + 56LL) + 2LL * (*(_DWORD *)(v8 + 16) >> 12);
      v10 = *(_DWORD *)(v8 + 16) & 0xFFF;
      result = 0;
      v11 = *(_QWORD *)(*a1 + 64LL) + 16LL * *(unsigned __int16 *)(v8 + 20);
      if ( v9 )
      {
        do
        {
          if ( v7 & *(_QWORD *)(v11 + 8 * result + 8) | v6 & *(_QWORD *)(v11 + 8 * result) )
            *(_QWORD *)(a1[1] + 8LL * (v10 >> 6)) |= 1LL << v10;
          v12 = *(__int16 *)(v9 + result);
          result += 2;
          v10 += v12;
        }
        while ( (_WORD)v12 );
      }
      v5 += 6;
    }
    while ( v3 != v5 );
  }
  return result;
}
