// Function: sub_159D9E0
// Address: 0x159d9e0
//
void __fastcall sub_159D9E0(__int64 a1)
{
  __int64 v2; // r12
  __int64 i; // rbx
  __int64 v4; // rdi

  v2 = 0;
  for ( i = *(_QWORD *)(a1 + 8); i; i = *(_QWORD *)(a1 + 8) )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v4 = sub_1648700(i);
        if ( *(_BYTE *)(v4 + 16) <= 0x10u )
        {
          if ( (unsigned __int8)sub_159D970(v4) )
            break;
        }
        v2 = i;
        i = *(_QWORD *)(i + 8);
        if ( !i )
          return;
      }
      if ( !v2 )
        break;
      i = *(_QWORD *)(v2 + 8);
      if ( !i )
        return;
    }
  }
}
