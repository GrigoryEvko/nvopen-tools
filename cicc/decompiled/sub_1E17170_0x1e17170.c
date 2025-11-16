// Function: sub_1E17170
// Address: 0x1e17170
//
__int64 __fastcall sub_1E17170(__int64 a1, int a2, unsigned int a3, unsigned int a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 result; // rax
  __int64 j; // r14
  __int64 v11; // rbx
  __int64 i; // r14
  unsigned int v13; // [rsp+Ch] [rbp-34h]

  v7 = a4;
  if ( (int)a3 <= 0 )
  {
    v11 = *(_QWORD *)(a1 + 32);
    result = 5LL * *(unsigned int *)(a1 + 40);
    for ( i = v11 + 40LL * *(unsigned int *)(a1 + 40); i != v11; v11 += 40 )
    {
      if ( !*(_BYTE *)v11 && a2 == *(_DWORD *)(v11 + 8) )
      {
        v13 = v7;
        result = sub_1E31150(v11, a3, v7, a5);
        v7 = v13;
      }
    }
  }
  else
  {
    if ( a4 )
      a3 = sub_38D6F10(a5 + 8, a3, a4);
    v8 = *(_QWORD *)(a1 + 32);
    result = 5LL * *(unsigned int *)(a1 + 40);
    for ( j = v8 + 40LL * *(unsigned int *)(a1 + 40); j != v8; v8 += 40 )
    {
      if ( !*(_BYTE *)v8 && a2 == *(_DWORD *)(v8 + 8) )
        result = sub_1E311F0(v8, a3, a5);
    }
  }
  return result;
}
