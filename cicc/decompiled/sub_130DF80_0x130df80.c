// Function: sub_130DF80
// Address: 0x130df80
//
unsigned __int64 __fastcall sub_130DF80(__int64 a1, unsigned __int64 a2, unsigned __int64 a3, unsigned int a4)
{
  unsigned __int64 result; // rax
  __int64 v6; // r15
  int v7; // r14d
  unsigned __int64 v8; // rdx
  unsigned __int64 v10; // [rsp+8h] [rbp-38h]

  result = *(unsigned int *)(a1 + 12);
  if ( (int)result > 0 )
  {
    result = (int)a4;
    v10 = (int)a4;
    v6 = a1 + 80;
    v7 = 0;
    do
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)(v6 + 13) )
          return result;
        result = sub_130DC10(*(_DWORD *)v6, *(_DWORD *)(v6 + 4), *(_DWORD *)(v6 + 8));
        if ( result >= a2 && result <= a3 )
          break;
LABEL_4:
        ++v7;
        v6 += 28;
        if ( *(_DWORD *)(a1 + 12) <= v7 )
          return result;
      }
      v8 = (result >> 12) - (((result & 0xFFF) == 0) - 1LL);
      if ( v10 < v8 )
      {
        *(_DWORD *)(v6 + 16) = v8;
        goto LABEL_4;
      }
      result = result << 9 >> 12;
      if ( v10 <= result )
        result = a4;
      ++v7;
      v6 += 28;
      *(_DWORD *)(v6 - 12) = result;
    }
    while ( *(_DWORD *)(a1 + 12) > v7 );
  }
  return result;
}
