// Function: sub_1B91520
// Address: 0x1b91520
//
void __fastcall sub_1B91520(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  unsigned __int8 *v9; // rsi
  __int64 v10; // rsi
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( a3 && *(_BYTE *)(a3 + 16) > 0x17u )
  {
    v5 = sub_15C70A0(a3 + 48);
    if ( v5
      && (v6 = sub_15F2060(a3), (unsigned __int8)sub_1626D30(v6))
      && (*(_BYTE *)(a3 + 16) != 78
       || (v8 = *(_QWORD *)(a3 - 24), *(_BYTE *)(v8 + 16))
       || (*(_BYTE *)(v8 + 33) & 0x20) == 0
       || (unsigned int)(*(_DWORD *)(v8 + 36) - 35) > 3) )
    {
      v7 = sub_1AFCF60(v5, *(_DWORD *)(a1 + 88) * *(_DWORD *)(a1 + 92));
    }
    else
    {
      v7 = v5;
    }
    sub_15C7080(v11, v7);
    if ( *a2 )
      sub_161E7C0((__int64)a2, *a2);
    v9 = (unsigned __int8 *)v11[0];
    *a2 = v11[0];
    if ( v9 )
      sub_1623210((__int64)v11, v9, (__int64)a2);
  }
  else
  {
    v10 = *a2;
    v11[0] = 0;
    if ( v10 )
    {
      sub_161E7C0((__int64)a2, v10);
      *a2 = v11[0];
    }
  }
}
