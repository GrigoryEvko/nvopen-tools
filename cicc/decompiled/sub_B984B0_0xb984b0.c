// Function: sub_B984B0
// Address: 0xb984b0
//
void __fastcall sub_B984B0(__int64 a1, unsigned __int8 (__fastcall *a2)(__int64, __int64, _QWORD), __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-28h]

  if ( *(_QWORD *)(a1 + 48) )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD))a2)(a3, 0) )
    {
      v5 = *(_QWORD *)(a1 + 48);
      v6 = 0;
      if ( v5 )
      {
        sub_B91220(a1 + 48, v5);
        *(_QWORD *)(a1 + 48) = v6;
      }
    }
  }
  sub_B98130(a1, a2, a3);
}
