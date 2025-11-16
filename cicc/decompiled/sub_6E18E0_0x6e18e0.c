// Function: sub_6E18E0
// Address: 0x6e18e0
//
void __fastcall sub_6E18E0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rcx

  v1 = *(_QWORD *)(a1 + 88);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 48);
    if ( v2 )
    {
      v3 = *(_QWORD *)(a1 + 88);
      while ( 1 )
      {
        *(_QWORD *)(v3 + 40) = v2;
        v3 = v2;
        if ( !*(_QWORD *)(v2 + 48) )
          break;
        v2 = *(_QWORD *)(v2 + 48);
      }
    }
    else
    {
      v2 = *(_QWORD *)(a1 + 88);
    }
    v4 = unk_4D03C48;
    unk_4D03C48 = v1;
    *(_QWORD *)(v2 + 40) = v4;
  }
}
