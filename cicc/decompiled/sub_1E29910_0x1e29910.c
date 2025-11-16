// Function: sub_1E29910
// Address: 0x1e29910
//
__int64 __fastcall sub_1E29910(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rax
  __int64 *v3; // r15
  __int64 *v4; // rbx
  __int64 v5; // r12

  v1 = 0;
  v2 = **(_QWORD **)(a1 + 32);
  v3 = *(__int64 **)(v2 + 72);
  v4 = *(__int64 **)(v2 + 64);
  if ( v4 != v3 )
  {
    while ( 1 )
    {
      v5 = *v4;
      if ( !sub_1DA1810(a1 + 56, *v4) )
      {
        if ( v1 && v5 != v1 )
          return 0;
        v1 = v5;
      }
      if ( v3 == ++v4 )
        return v1;
    }
  }
  return 0;
}
