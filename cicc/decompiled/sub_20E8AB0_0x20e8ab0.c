// Function: sub_20E8AB0
// Address: 0x20e8ab0
//
__int64 __fastcall sub_20E8AB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v3; // r12d
  __int64 v4; // rbx
  int v5; // edx
  __int64 result; // rax
  int v7; // edx
  int v8; // r8d
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rsi
  unsigned int i; // eax
  __int64 v12; // rsi
  unsigned int v13; // eax

  v2 = sub_20E8320(a1, *(unsigned __int16 *)(a2 + 6));
  v3 = *(_DWORD *)(a1 + 8);
  v4 = v2;
  sub_20E8610(a1, v3);
  v5 = *(_DWORD *)(a1 + 56);
  result = 0;
  if ( v5 )
  {
    v7 = v5 - 1;
    v8 = 1;
    v9 = ((((unsigned int)(37 * v4) | ((unsigned __int64)(37 * v3) << 32))
         - 1
         - ((unsigned __int64)(unsigned int)(37 * v4) << 32)) >> 22)
       ^ (((unsigned int)(37 * v4) | ((unsigned __int64)(37 * v3) << 32))
        - 1
        - ((unsigned __int64)(unsigned int)(37 * v4) << 32));
    v10 = ((9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13)))) >> 15)
        ^ (9 * (((v9 - 1 - (v9 << 13)) >> 8) ^ (v9 - 1 - (v9 << 13))));
    for ( i = v7 & (((v10 - 1 - (v10 << 27)) >> 31) ^ (v10 - 1 - ((_DWORD)v10 << 27))); ; i = v7 & v13 )
    {
      v12 = *(_QWORD *)(a1 + 40) + 24LL * i;
      if ( v3 == *(_DWORD *)v12 && v4 == *(_QWORD *)(v12 + 8) )
        return 1;
      if ( *(_DWORD *)v12 == -1 && *(_QWORD *)(v12 + 8) == -1 )
        break;
      v13 = v8 + i;
      ++v8;
    }
    return 0;
  }
  return result;
}
