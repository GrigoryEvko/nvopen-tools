// Function: sub_1D12510
// Address: 0x1d12510
//
char *__fastcall sub_1D12510(__int64 a1, __int64 a2)
{
  char *result; // rax
  __int64 v3; // rbx
  __int64 i; // r13
  _BYTE *v6; // rsi
  unsigned __int64 v7; // r12
  unsigned __int64 v8[7]; // [rsp+18h] [rbp-38h] BYREF

  result = (char *)(a1 + 344);
  v3 = *(_QWORD *)(a2 + 112);
  for ( i = v3 + 16LL * *(unsigned int *)(a2 + 120); v3 != i; v3 += 16 )
  {
    v7 = *(_QWORD *)v3 & 0xFFFFFFFFFFFFFFF8LL;
    --*(_DWORD *)(v7 + 208);
    v8[0] = v7;
    if ( (*(_BYTE *)(a2 + 236) & 1) == 0 )
      sub_1F01DD0(a2);
    sub_1F01F20(v7, (unsigned int)(*(_DWORD *)(a2 + 240) + *(_DWORD *)(v3 + 12)));
    result = (char *)*(unsigned int *)(v8[0] + 208);
    if ( !(_DWORD)result && v8[0] != a1 + 344 )
    {
      v6 = *(_BYTE **)(a1 + 680);
      if ( v6 == *(_BYTE **)(a1 + 688) )
      {
        result = sub_1CFD630(a1 + 672, v6, v8);
      }
      else
      {
        if ( v6 )
        {
          *(_QWORD *)v6 = v8[0];
          v6 = *(_BYTE **)(a1 + 680);
        }
        *(_QWORD *)(a1 + 680) = v6 + 8;
      }
    }
  }
  return result;
}
