// Function: sub_12781D0
// Address: 0x12781d0
//
void __fastcall sub_12781D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 i; // r15
  _BYTE *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 )
  {
    for ( i = 0; i != a1; ++i )
    {
      while ( 1 )
      {
        v7 = sub_1643330(a3);
        v6 = *(_BYTE **)(a2 + 8);
        v8 = v7;
        if ( v6 != *(_BYTE **)(a2 + 16) )
          break;
        ++i;
        sub_1278040(a2, v6, &v8);
        if ( a1 == i )
          return;
      }
      if ( v6 )
      {
        *(_QWORD *)v6 = v7;
        v6 = *(_BYTE **)(a2 + 8);
      }
      *(_QWORD *)(a2 + 8) = v6 + 8;
    }
  }
}
