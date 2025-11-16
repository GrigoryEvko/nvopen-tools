// Function: sub_1E6A2A0
// Address: 0x1e6a2a0
//
void __fastcall sub_1E6A2A0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax

  if ( a2 < 0 )
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v2 = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8LL * (unsigned int)a2);
  if ( v2 )
  {
    if ( (*(_BYTE *)(v2 + 3) & 0x10) != 0 )
    {
      while ( 1 )
      {
        v2 = *(_QWORD *)(v2 + 32);
        if ( !v2 )
          break;
        if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 )
          goto LABEL_5;
      }
    }
    else
    {
LABEL_5:
      v3 = *(_QWORD *)(v2 + 16);
      while ( 1 )
      {
        v2 = *(_QWORD *)(v2 + 32);
        if ( !v2 )
          break;
        if ( (*(_BYTE *)(v2 + 3) & 0x10) == 0 && v3 != *(_QWORD *)(v2 + 16) )
        {
          if ( **(_WORD **)(v3 + 16) == 12 )
            sub_1E310D0(*(_QWORD *)(v3 + 32), 0);
          goto LABEL_5;
        }
      }
      if ( **(_WORD **)(v3 + 16) == 12 )
        sub_1E310D0(*(_QWORD *)(v3 + 32), 0);
    }
  }
}
