// Function: sub_399C630
// Address: 0x399c630
//
__int64 __fastcall sub_399C630(__int64 a1)
{
  __int64 *v1; // r15
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v5; // r14
  __int64 v6; // rbx
  __int64 v7; // rdi

  v1 = *(__int64 **)(a1 + 3864);
  result = *(unsigned int *)(a1 + 3872);
  for ( i = &v1[result]; i != v1; result = sub_39C95F0(v7, v5) )
  {
    while ( 1 )
    {
      v5 = *v1;
      v6 = sub_3999410(a1, *(_QWORD *)(*v1 + 8 * (5LL - *(unsigned int *)(*v1 + 8))));
      result = sub_39C95F0(v6, v5);
      v7 = *(_QWORD *)(v6 + 616);
      if ( v7 )
      {
        result = *(_QWORD *)(v6 + 80);
        if ( *(_BYTE *)(result + 48) )
          break;
      }
      if ( i == ++v1 )
        return result;
    }
    ++v1;
  }
  return result;
}
