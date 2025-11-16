// Function: sub_1524C40
// Address: 0x1524c40
//
__int64 __fastcall sub_1524C40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rsi

  v5 = sub_16982C0(a1, a2, a3, a4);
  v6 = a2 + 8;
  if ( *(_QWORD *)(a2 + 8) == v5 )
    sub_169D930(a1, v6);
  else
    sub_169D7E0(a1, v6);
  return a1;
}
