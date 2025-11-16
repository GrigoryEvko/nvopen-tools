// Function: sub_72F310
// Address: 0x72f310
//
__int64 __fastcall sub_72F310(__int64 a1, int a2)
{
  __int64 i; // rax
  unsigned int v3; // r8d
  __int64 *v5; // rax

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v3 = 0;
  if ( *(_BYTE *)(a1 + 174) == 1 )
  {
    v3 = 1;
    v5 = **(__int64 ***)(i + 168);
    if ( v5 )
    {
      if ( (v5[4] & 4) != 0 )
      {
        if ( !a2 && (*(_BYTE *)(a1 + 195) & 8) == 0 )
        {
          do
          {
            if ( !v5[5] && (v5[4] & 0x10) == 0 )
              return 0;
            v5 = (__int64 *)*v5;
          }
          while ( v5 );
          return 1;
        }
      }
      else
      {
        v3 = 0;
        if ( (*((_BYTE *)v5 + 33) & 1) != 0 )
          return *v5 == 0;
      }
    }
  }
  return v3;
}
