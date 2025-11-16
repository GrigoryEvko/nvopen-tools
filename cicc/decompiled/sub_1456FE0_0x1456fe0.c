// Function: sub_1456FE0
// Address: 0x1456fe0
//
__int64 __fastcall sub_1456FE0(__int64 a1)
{
  __int64 v1; // r8
  __int64 *v2; // rax
  __int64 *v3; // rdx

  v1 = a1;
  if ( *(_WORD *)(a1 + 24) )
  {
    v1 = 0;
    if ( *(_WORD *)(a1 + 24) == 5 )
    {
      v2 = *(__int64 **)(a1 + 32);
      v3 = &v2[*(_QWORD *)(a1 + 40)];
      if ( v2 != v3 )
      {
        while ( 1 )
        {
          v1 = *v2;
          if ( !*(_WORD *)(*v2 + 24) )
            break;
          if ( v3 == ++v2 )
            return 0;
        }
      }
    }
  }
  return v1;
}
