// Function: sub_708EB0
// Address: 0x708eb0
//
void __fastcall sub_708EB0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // rdi

  if ( a1 )
  {
    v1 = a1;
    do
    {
      while ( 1 )
      {
        if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u && !(unsigned int)sub_736DD0(v1) )
        {
          v2 = *(_QWORD *)(*(_QWORD *)(v1 + 168) + 152LL);
          if ( v2 )
          {
            if ( (*(_BYTE *)(v2 + 29) & 0x20) == 0 )
              break;
          }
        }
        v1 = *(_QWORD *)(v1 + 112);
        if ( !v1 )
          return;
      }
      sub_708DA0(v2);
      v1 = *(_QWORD *)(v1 + 112);
    }
    while ( v1 );
  }
}
