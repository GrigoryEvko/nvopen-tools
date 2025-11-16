// Function: sub_255E180
// Address: 0x255e180
//
__int64 *__fastcall sub_255E180(__int64 a1, __int64 *a2)
{
  int v2; // eax
  __int64 v3; // r9
  __int64 v4; // rsi
  int v5; // ecx
  unsigned int v6; // eax
  int v7; // edi
  __int64 *v8; // r8
  __int64 v9; // rdx

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *a2;
    v5 = v2 - 1;
    v6 = (v2 - 1) & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
    v7 = 1;
    v8 = (__int64 *)(v3 + 104LL * v6);
    v9 = *v8;
    if ( v4 == *v8 )
      return v8;
    while ( v9 != -4096 )
    {
      v6 = v5 & (v7 + v6);
      v8 = (__int64 *)(v3 + 104LL * v6);
      v9 = *v8;
      if ( *v8 == v4 )
        return v8;
      ++v7;
    }
  }
  return 0;
}
