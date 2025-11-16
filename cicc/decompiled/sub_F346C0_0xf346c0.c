// Function: sub_F346C0
// Address: 0xf346c0
//
__int64 __fastcall sub_F346C0(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  _QWORD *v5; // rax
  unsigned __int16 v6; // dx
  unsigned __int16 v7; // r14
  _QWORD *v8; // r13
  __int64 v10; // rsi
  __int64 v11; // r14
  __int64 v12; // rsi
  unsigned __int8 *v13; // rsi
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  if ( !*((_QWORD *)a3 + 6) )
  {
    if ( !*(_QWORD *)a2 )
      BUG();
    v10 = *(_QWORD *)(*(_QWORD *)a2 + 24LL);
    v11 = (__int64)(a3 + 48);
    v14[0] = v10;
    if ( v10 )
    {
      sub_B96E90((__int64)v14, v10, 1);
      v12 = *((_QWORD *)a3 + 6);
      if ( v12 )
        sub_B91220(v11, v12);
      v13 = (unsigned __int8 *)v14[0];
      *((_QWORD *)a3 + 6) = v14[0];
      if ( v13 )
        sub_B976B0((__int64)v14, v13, v11);
    }
  }
  v5 = sub_B44240(a3, a1, *(unsigned __int64 **)a2, *(_QWORD *)(a2 + 8));
  v7 = v6;
  v8 = v5;
  sub_F34640(a2, a3);
  *(_QWORD *)a2 = v8;
  *(_WORD *)(a2 + 8) = v7;
  return v7;
}
