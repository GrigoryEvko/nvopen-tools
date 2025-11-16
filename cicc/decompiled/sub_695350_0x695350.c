// Function: sub_695350
// Address: 0x695350
//
__int64 __fastcall sub_695350(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r12
  __int64 v5; // r12
  __int64 v6; // rbx
  _BYTE v7[160]; // [rsp+0h] [rbp-220h] BYREF
  _BYTE v8[19]; // [rsp+A0h] [rbp-180h] BYREF
  char v9; // [rsp+B3h] [rbp-16Dh]

  sub_6E1E00((unsigned int)(a3 == 0) + 4, v7, 0, 1);
  if ( *(_BYTE *)(a1 + 24) == 2 )
  {
    v5 = *(_QWORD *)(a1 + 56);
    v6 = *(_QWORD *)(v5 + 144);
    *(_QWORD *)(v5 + 144) = 0;
    sub_6E6A50(v5, v8);
    if ( sub_694910(v8) )
      v9 |= 0x10u;
    *(_QWORD *)(v5 + 144) = v6;
  }
  else
  {
    sub_6E7170(a1, v8);
  }
  sub_843D70(v8, a2, 0, 310);
  v3 = sub_6F6F40(v8, 0);
  sub_6E2B30(v8, 0);
  return v3;
}
