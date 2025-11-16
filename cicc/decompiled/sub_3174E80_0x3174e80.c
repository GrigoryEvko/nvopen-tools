// Function: sub_3174E80
// Address: 0x3174e80
//
__int64 __fastcall sub_3174E80(__int64 a1, __int64 a2, __int64 a3, unsigned int a4, __int64 *a5)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 i; // r10
  unsigned int v10; // ecx
  __int64 v11; // rdx
  unsigned int v12; // eax
  bool v13; // cc
  __int64 *v15; // [rsp+8h] [rbp-38h] BYREF

  v7 = a2;
  v8 = (a3 - 1) / 2;
  if ( a2 < v8 )
  {
    for ( i = a2; ; i = a2 )
    {
      a2 = 2 * (i + 1);
      v10 = *(_DWORD *)(a1 + 8 * (i + 1));
      v11 = 176LL * v10;
      v12 = *(_DWORD *)(*a5 + 176LL * *(unsigned int *)(a1 + 8 * (i + 1) - 4) + 104);
      v13 = *(_DWORD *)(*a5 + v11 + 104) <= v12;
      if ( *(_DWORD *)(*a5 + v11 + 104) == v12 )
        v13 = v10 <= *(_DWORD *)(a1 + 8 * (i + 1) - 4);
      if ( !v13 )
      {
        --a2;
        v10 = *(_DWORD *)(a1 + 4 * a2);
      }
      *(_DWORD *)(a1 + 4 * i) = v10;
      if ( a2 >= v8 )
        break;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == a2 )
  {
    *(_DWORD *)(a1 + 4 * a2) = *(_DWORD *)(a1 + 4 * (2 * a2 + 2) - 4);
    a2 = 2 * a2 + 1;
  }
  v15 = a5;
  return sub_3174DD0(a1, a2, v7, a4, &v15);
}
