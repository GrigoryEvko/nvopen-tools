// Function: sub_26ECEB0
// Address: 0x26eceb0
//
void __fastcall sub_26ECEB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rsi

  v2 = a2 + 24;
  v3 = *(_QWORD *)(a2 + 32);
  if ( v3 != a2 + 24 )
  {
    do
    {
      v4 = v3 - 56;
      if ( !v3 )
        v4 = 0;
      sub_26ECD90(a1, v4);
      v3 = *(_QWORD *)(v3 + 8);
    }
    while ( v2 != v3 );
  }
}
