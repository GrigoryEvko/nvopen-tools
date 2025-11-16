// Function: sub_D97190
// Address: 0xd97190
//
__int64 __fastcall sub_D97190(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 *v4; // rbx
  __int64 v5; // rax
  __int64 *i; // r14
  __int64 v7; // r13
  __int16 v8; // ax
  __int16 v9; // ax

  v2 = a2;
  if ( *(_BYTE *)(sub_D95540(a2) + 8) == 14 )
  {
    while ( 1 )
    {
      v8 = *(_WORD *)(v2 + 24);
      if ( v8 == 8 )
      {
        do
        {
          v2 = **(_QWORD **)(v2 + 32);
          v9 = *(_WORD *)(v2 + 24);
        }
        while ( v9 == 8 );
        if ( v9 != 5 )
          return v2;
      }
      else if ( v8 != 5 )
      {
        return v2;
      }
      v4 = *(__int64 **)(v2 + 32);
      v5 = *(_QWORD *)(v2 + 40);
      v2 = 0;
      for ( i = &v4[v5]; i != v4; ++v4 )
      {
        v7 = *v4;
        if ( *(_BYTE *)(sub_D95540(*v4) + 8) == 14 )
          v2 = v7;
      }
    }
  }
  return v2;
}
