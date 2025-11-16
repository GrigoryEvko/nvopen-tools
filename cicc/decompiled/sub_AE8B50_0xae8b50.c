// Function: sub_AE8B50
// Address: 0xae8b50
//
void __fastcall sub_AE8B50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rax
  _QWORD v8[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( !*(_BYTE *)(a3 + 32) )
  {
    v5 = sub_B12000(a3 + 72);
    sub_AE8A40(a1, a2, v5);
  }
  v6 = *(_QWORD *)(a3 + 24);
  v8[0] = v6;
  if ( v6 )
    sub_B96E90(v8, v6, 1);
  v7 = sub_B10CD0(v8);
  sub_AE8180(a1, a2, v7);
  if ( v8[0] )
    sub_B91220(v8);
}
