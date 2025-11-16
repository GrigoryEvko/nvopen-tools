// Function: sub_7342B0
// Address: 0x7342b0
//
void __fastcall sub_7342B0(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  __int64 v3; // r15

  v1 = *(_QWORD *)(a1 + 32);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 32);
    if ( v2 )
    {
      v3 = *(_QWORD *)(v2 + 32);
      if ( v3 )
      {
        if ( *(_QWORD *)(v3 + 32) )
        {
          sub_7342B0();
          *(_QWORD *)(v3 + 32) = 0;
        }
        *(_QWORD *)(v3 + 24) = 0;
        sub_7340D0(v3, 0, 0);
        *(_QWORD *)(v2 + 32) = 0;
      }
      *(_QWORD *)(v2 + 24) = 0;
      sub_7340D0(v2, 0, 0);
      *(_QWORD *)(v1 + 32) = 0;
    }
    *(_QWORD *)(v1 + 24) = 0;
    sub_7340D0(v1, 0, 0);
    *(_QWORD *)(a1 + 32) = 0;
  }
  *(_QWORD *)(a1 + 24) = 0;
  sub_7340D0(a1, 0, 0);
}
