// Function: sub_2EA6550
// Address: 0x2ea6550
//
__int64 __fastcall sub_2EA6550(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 *v3; // r14
  __int64 *i; // rbx
  __int64 v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rdx

  v1 = sub_2EA49F0(a1);
  v2 = v1;
  if ( !v1 )
    return v2;
  v3 = *(__int64 **)(v1 + 112);
  for ( i = &v3[*(unsigned int *)(v1 + 120)]; i != v3; ++v3 )
  {
    v5 = *v3;
    if ( *(_BYTE *)(a1 + 84) )
    {
      v6 = *(_QWORD **)(a1 + 64);
      v7 = &v6[*(unsigned int *)(a1 + 76)];
      if ( v6 == v7 )
        return v2;
      while ( v5 != *v6 )
      {
        if ( v7 == ++v6 )
          return v2;
      }
    }
    else if ( !sub_C8CA60(a1 + 56, v5) )
    {
      return v2;
    }
  }
  return sub_2EA43D0(a1);
}
