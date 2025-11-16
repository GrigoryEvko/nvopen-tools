// Function: sub_2FE4070
// Address: 0x2fe4070
//
__int64 __fastcall sub_2FE4070(__int64 a1, unsigned __int16 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 v5; // rax

  v5 = 1;
  if ( a2 != 1 )
  {
    a5 = 0;
    if ( !a2 )
      return 0;
    v5 = a2;
    if ( !*(_QWORD *)(a1 + 8LL * a2 + 112) )
      return 0;
  }
  LOBYTE(a5) = *(_BYTE *)(a1 + 500 * v5 + 6614) == 0;
  return a5;
}
