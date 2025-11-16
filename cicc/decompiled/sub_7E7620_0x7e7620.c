// Function: sub_7E7620
// Address: 0x7e7620
//
void __fastcall sub_7E7620(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rbx

  v2 = *(_BYTE *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 16);
  if ( v2 != 11 || v3 )
  {
    while ( 1 )
    {
LABEL_5:
      *(_QWORD *)(a1 + 16) = 0;
      if ( v2 != 11 || *(_QWORD *)(a1 + 72) || **(_DWORD **)(a1 + 80) )
      {
        sub_7E6810(a1, a2, 1);
        if ( !v3 )
          return;
      }
      else if ( !v3 )
      {
        return;
      }
      v2 = *(_BYTE *)(v3 + 40);
      a1 = v3;
      v3 = *(_QWORD *)(v3 + 16);
    }
  }
  a1 = *(_QWORD *)(a1 + 72);
  if ( a1 )
  {
    v2 = *(_BYTE *)(a1 + 40);
    v3 = *(_QWORD *)(a1 + 16);
    goto LABEL_5;
  }
}
