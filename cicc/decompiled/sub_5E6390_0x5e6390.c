// Function: sub_5E6390
// Address: 0x5e6390
//
void __fastcall sub_5E6390(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rsi

  if ( a2 )
    a1 = *(_QWORD *)(a2 + 40);
  if ( (*(_BYTE *)(a1 + 176) & 0x50) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 168);
    v4 = *(_QWORD *)(v3 + 80);
    if ( v4 )
    {
      if ( a2 )
        v4 = sub_8E5650(*(_QWORD *)(v3 + 80));
      *(_BYTE *)(v4 + 96) |= 8u;
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 24LL);
      if ( v5 )
      {
        if ( a2 )
LABEL_9:
          v5 = sub_8E5650(v5);
        if ( v4 != v5 )
        {
          do
          {
            v6 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 40) + 168LL) + 24LL);
            if ( !v6 )
              break;
            v7 = v5;
            *(_BYTE *)(v5 + 96) |= 8u;
            v5 = v6;
            if ( v7 )
              goto LABEL_9;
          }
          while ( v4 != v6 );
        }
      }
    }
  }
}
