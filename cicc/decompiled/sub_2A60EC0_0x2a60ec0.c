// Function: sub_2A60EC0
// Address: 0x2a60ec0
//
char __fastcall sub_2A60EC0(__int64 a1, __int64 a2, char a3)
{
  unsigned __int64 v4; // rsi

  if ( !a1 )
    return 0;
  v4 = *(_QWORD *)(a1 + 56);
  if ( a3 )
    return sub_D84450(a2, v4) ^ 1;
  else
    return sub_D84440(a2, v4);
}
