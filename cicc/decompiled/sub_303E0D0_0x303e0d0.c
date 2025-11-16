// Function: sub_303E0D0
// Address: 0x303e0d0
//
__int64 __fastcall sub_303E0D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int16 v9; // bx
  __int64 v10; // rsi
  __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-50h]
  unsigned int v14; // [rsp+10h] [rbp-40h] BYREF
  __int64 v15; // [rsp+18h] [rbp-38h]

  v9 = *(_WORD *)(a2 + 96);
  v10 = *(_QWORD *)(a2 + 104);
  LOWORD(v14) = v9;
  v15 = v10;
  if ( v9 == 2 )
    return sub_303DF80(a1, a2, a3, a4, a5, a6);
  if ( (unsigned __int8)sub_307AB50(v14, v10) == 1 || v9 == 37 )
  {
    v13 = *(_QWORD *)(a2 + 112);
    v12 = sub_2E79000(*(__int64 **)(a4 + 40));
    if ( !(unsigned __int8)sub_2FEBAB0(a1, *(_QWORD *)(a4 + 64), v12, v14, v15, v13, 0) )
      return sub_3463A90(a1, a2, a4);
  }
  if ( (unsigned __int8)sub_307AB50(v14, v15) == 1 || v9 == 37 )
    return 0;
  if ( v9 )
  {
    if ( (unsigned __int16)(v9 - 17) > 0xD3u )
      return 0;
  }
  else if ( !sub_30070B0((__int64)&v14) )
  {
    return 0;
  }
  return sub_303D690(a1, a2, a3, a4);
}
