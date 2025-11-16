// Function: sub_E02AA0
// Address: 0xe02aa0
//
void __fastcall sub_E02AA0(__int64 a1)
{
  __int64 i; // rbx
  __int64 v2; // rdi
  __int64 j; // r13
  __int64 v4; // r12
  __int64 v5; // rsi

  for ( i = *(_QWORD *)(a1 + 16); i; i = *(_QWORD *)(i + 8) )
  {
    v2 = *(_QWORD *)(i + 24);
    if ( *(_BYTE *)v2 == 6 )
    {
      sub_E02AA0();
    }
    else if ( *(_BYTE *)v2 == 5 && *(_WORD *)(v2 + 2) == 47 )
    {
      for ( j = *(_QWORD *)(v2 + 16); j; j = *(_QWORD *)(j + 8) )
      {
        v4 = *(_QWORD *)(j + 24);
        if ( *(_BYTE *)v4 != 5 )
          break;
        if ( *(_WORD *)(v4 + 2) != 15 )
          break;
        v5 = sub_AD64C0(*(_QWORD *)(v4 + 8), 0, 0);
        sub_BD84E0(v4, v5);
      }
    }
  }
}
