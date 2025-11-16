// Function: sub_1EAABB0
// Address: 0x1eaabb0
//
__int64 __fastcall sub_1EAABB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r12
  bool v4; // zf
  _BYTE *v5; // rsi
  __int64 result; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a2 + 112);
  for ( i = v2 + 16LL * *(unsigned int *)(a2 + 120); v2 != i; v2 += 16 )
  {
    while ( 1 )
    {
      result = *(_QWORD *)v2 ^ 6LL;
      v7 = *(_QWORD *)v2 & 0xFFFFFFFFFFFFFFF8LL;
      v8[0] = v7;
      if ( (result & 6) != 0 || *(_DWORD *)(v2 + 8) <= 3u )
        break;
      v2 += 16;
      --*(_DWORD *)(v7 + 216);
      if ( v2 == i )
        return result;
    }
    v4 = (*(_DWORD *)(v7 + 208))-- == 1;
    if ( v4 && v7 != a1 + 344 )
    {
      v5 = *(_BYTE **)(a1 + 2192);
      if ( v5 == *(_BYTE **)(a1 + 2200) )
      {
        result = (__int64)sub_1CFD630(a1 + 2184, v5, v8);
      }
      else
      {
        if ( v5 )
        {
          *(_QWORD *)v5 = v7;
          v5 = *(_BYTE **)(a1 + 2192);
        }
        *(_QWORD *)(a1 + 2192) = v5 + 8;
      }
    }
  }
  return result;
}
