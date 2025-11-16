// Function: sub_8C3170
// Address: 0x8c3170
//
void __fastcall sub_8C3170(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u )
      {
        v2 = *(_QWORD *)(*(_QWORD *)(v1 + 168) + 152LL);
        if ( v2 )
        {
          if ( (*(_BYTE *)(v2 + 29) & 0x20) == 0 )
            sub_8C3020(v2);
        }
      }
      v1 = *(_QWORD *)(v1 + 112);
    }
    while ( v1 );
  }
}
