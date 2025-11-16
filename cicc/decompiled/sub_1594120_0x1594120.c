// Function: sub_1594120
// Address: 0x1594120
//
__int64 __fastcall sub_1594120(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi

  v4 = *(_QWORD *)(a1 + 32);
  if ( v4 != *(_QWORD *)(a2 + 8) )
    return 0;
  v6 = sub_16982C0(a1, a2, a3, a4);
  v7 = a1 + 32;
  v8 = a2 + 8;
  if ( v4 == v6 )
    return sub_169CB90(v7, v8);
  else
    return sub_1698510(v7, v8);
}
