// Function: sub_7E1270
// Address: 0x7e1270
//
void __fastcall sub_7E1270(__int64 a1)
{
  __int64 v1; // rbx
  char v2; // al
  __int64 v3; // rax
  _BYTE *v4; // rdi
  __int64 i; // rbx

  if ( a1 )
  {
    v1 = a1;
    while ( (*(_BYTE *)(v1 - 8) & 1) != 0 )
    {
      v4 = *(_BYTE **)(v1 + 160);
      if ( !v4 || (v4[156] & 1) != 0 )
      {
        v2 = *(_BYTE *)(v1 + 173);
        if ( v2 == 10 )
          goto LABEL_12;
      }
      else
      {
        sub_7E1230(v4, 0, 0, 0);
        v2 = *(_BYTE *)(v1 + 173);
        if ( v2 == 10 )
        {
LABEL_12:
          for ( i = *(_QWORD *)(v1 + 176); i; i = *(_QWORD *)(i + 120) )
            sub_7E1270(i);
          return;
        }
      }
      if ( v2 == 6 && (unsigned __int8)(*(_BYTE *)(v1 + 176) - 2) <= 1u )
      {
        v3 = *(_QWORD *)(v1 + 184);
        if ( v3 )
        {
          *(_BYTE *)(v3 + 168) |= 0x10u;
          v1 = *(_QWORD *)(v1 + 184);
          if ( v1 )
            continue;
        }
      }
      return;
    }
  }
}
