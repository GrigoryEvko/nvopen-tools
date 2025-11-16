// Function: sub_816050
// Address: 0x816050
//
void __fastcall sub_816050(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r12
  __int64 v3; // rax

  if ( a1 )
  {
    v1 = a1;
    do
    {
      if ( !(unsigned int)sub_736DD0(v1) )
      {
        sub_815C30(v1);
        if ( (unsigned __int8)(*(_BYTE *)(v1 + 140) - 9) <= 2u )
        {
          v2 = *(_QWORD *)(v1 + 168);
          v3 = *(_QWORD *)(v2 + 152);
          if ( v3 )
          {
            if ( (*(_BYTE *)(v3 + 29) & 0x20) == 0 )
              sub_816050(*(_QWORD *)(v3 + 104));
          }
          sub_816050(*(_QWORD *)(v2 + 216));
        }
      }
      v1 = *(_QWORD *)(v1 + 112);
    }
    while ( v1 );
  }
}
