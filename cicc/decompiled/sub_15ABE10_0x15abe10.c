// Function: sub_15ABE10
// Address: 0x15abe10
//
void __fastcall sub_15ABE10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  char v6; // dl
  __int64 v7; // rsi
  __int64 v8; // rax
  _QWORD v9[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( *(_BYTE *)(a3 + 16) == 78 )
  {
    v5 = *(_QWORD *)(a3 - 24);
    if ( !*(_BYTE *)(v5 + 16) )
    {
      v6 = *(_BYTE *)(v5 + 33);
      if ( (v6 & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v5 + 36) == 36 )
        {
          sub_15ABCE0(a1, a2, a3);
        }
        else if ( (v6 & 0x20) != 0 && *(_DWORD *)(v5 + 36) == 38 )
        {
          sub_15ABE00(a1, a2, a3);
        }
      }
    }
  }
  v7 = *(_QWORD *)(a3 + 48);
  v9[0] = v7;
  if ( v7 )
  {
    sub_1623A60(v9, v7, 2);
    if ( v9[0] )
    {
      v8 = sub_15C70A0(v9);
      sub_15AB850(a1, a2, v8);
      if ( v9[0] )
        sub_161E7C0(v9);
    }
  }
}
