// Function: sub_85E780
// Address: 0x85e780
//
void __fastcall sub_85E780(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rsi
  __int64 v3; // rax

  v1 = *(__int64 **)(a1 + 56);
  if ( v1 )
  {
    v2 = 0;
    do
    {
      while ( 1 )
      {
        if ( *((_BYTE *)v1 + 8) != 6 )
          sub_721090();
        v3 = v1[2];
        if ( (*(_BYTE *)(*(_QWORD *)(v3 + 168) + 109LL) & 0x20) != 0 )
          break;
        v1 = (__int64 *)*v1;
        if ( !v1 )
          return;
      }
      ++v2;
      for ( ; *(_BYTE *)(v3 + 140) == 12; v3 = *(_QWORD *)(v3 + 160) )
        ;
      *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v3 + 96LL) + 168LL) = v2;
      v1 = (__int64 *)*v1;
    }
    while ( v1 );
  }
}
