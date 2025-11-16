// Function: sub_13191B0
// Address: 0x13191b0
//
void __fastcall sub_13191B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int64 v3; // rbx
  __int64 v4; // rsi
  __int64 v5; // rsi
  __int64 v6; // rsi

  v5 = a2 + 10648;
  if ( *(_BYTE *)(v5 + 17) )
  {
    v6 = v5 + 62264;
    if ( *(_QWORD *)(v6 + 64) )
    {
      v2 = v6;
      v3 = 0;
      do
      {
        v4 = 9 * v3++;
        sub_130B000(a1, *(_QWORD *)(v2 + 104) + 16 * v4);
      }
      while ( *(_QWORD *)(v2 + 64) > v3 );
    }
    else
    {
      nullsub_2017();
    }
  }
}
