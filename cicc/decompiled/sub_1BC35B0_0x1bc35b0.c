// Function: sub_1BC35B0
// Address: 0x1bc35b0
//
void __fastcall sub_1BC35B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v7; // rsi

  v7 = a1[1];
  if ( v7 == a1[2] )
  {
    sub_1BC3330(a1, (char *)v7, a2, a4);
  }
  else
  {
    if ( v7 )
    {
      *(_QWORD *)v7 = *(_QWORD *)a2;
      *(_QWORD *)(v7 + 8) = v7 + 24;
      *(_QWORD *)(v7 + 16) = 0x200000000LL;
      if ( *(_DWORD *)(a2 + 16) )
        sub_1BB9C60(v7 + 8, (char **)(a2 + 8), a2, a4, a2 + 8, a6);
      v7 = a1[1];
    }
    a1[1] = v7 + 40;
  }
}
