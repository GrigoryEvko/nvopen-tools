// Function: sub_109CEB0
// Address: 0x109ceb0
//
__int64 __fastcall sub_109CEB0(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  int v9; // eax
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_QWORD *)(v4 + 48);
  v11[0] = v5;
  if ( !v5 )
  {
    v7 = *(_QWORD *)(a2 + 48);
    v6 = a2 + 48;
    if ( !v7 )
      goto LABEL_7;
    goto LABEL_3;
  }
  v6 = a2 + 48;
  sub_B96E90((__int64)v11, v5, 1);
  v7 = *(_QWORD *)(a2 + 48);
  if ( v7 )
LABEL_3:
    sub_B91220(v6, v7);
  v8 = (unsigned __int8 *)v11[0];
  *(_QWORD *)(a2 + 48) = v11[0];
  if ( v8 )
    sub_B976B0((__int64)v11, v8, v6);
  v4 = *(_QWORD *)(a1 + 8);
LABEL_7:
  v9 = sub_B45210(v4);
  return sub_B45150(a2, v9);
}
