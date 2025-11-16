// Function: sub_295CC00
// Address: 0x295cc00
//
__int64 __fastcall sub_295CC00(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  __int64 v4; // r8
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // rsi
  int v9; // edx
  int v10; // r8d
  unsigned int v11; // eax
  __int64 v12; // rdi

  if ( !*(_DWORD *)(a1 + 16) )
  {
    v2 = *(unsigned int *)(a1 + 40);
    v3 = *(_QWORD **)(a1 + 32);
    LOBYTE(v4) = &v3[v2] != sub_2957650(v3, (__int64)&v3[v2], a2);
    return v4;
  }
  v6 = *(_DWORD *)(a1 + 24);
  v7 = *a2;
  v8 = *(_QWORD *)(a1 + 8);
  if ( v6 )
  {
    v9 = v6 - 1;
    v10 = 1;
    v11 = (v6 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v12 = *(_QWORD *)(v8 + 8LL * v11);
    if ( v7 == v12 )
      return 1;
    while ( v12 != -4096 )
    {
      v11 = v9 & (v10 + v11);
      v12 = *(_QWORD *)(v8 + 8LL * v11);
      if ( v7 == v12 )
        return 1;
      ++v10;
    }
  }
  return 0;
}
