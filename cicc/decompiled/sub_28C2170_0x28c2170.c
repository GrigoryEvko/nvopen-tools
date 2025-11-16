// Function: sub_28C2170
// Address: 0x28c2170
//
__int64 __fastcall sub_28C2170(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 *a4)
{
  __int64 v6; // r12
  int v7; // eax
  __int64 v8; // rsi
  __int64 *v9; // r15
  __int64 v11; // rsi
  unsigned __int8 *v12; // rsi
  __int64 v13[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v6 = (__int64)sub_28C1F40(a1, a2, (__int64)a4);
  if ( v6 )
  {
    v7 = *a4;
    if ( v7 == 42 )
    {
      v14 = 257;
      v6 = sub_B504D0(13, v6, a3, (__int64)v13, (__int64)(a4 + 24), 0);
    }
    else
    {
      if ( v7 != 46 )
        BUG();
      v14 = 257;
      v6 = sub_B504D0(17, v6, a3, (__int64)v13, (__int64)(a4 + 24), 0);
    }
    v8 = *((_QWORD *)a4 + 6);
    v9 = (__int64 *)(v6 + 48);
    v13[0] = v8;
    if ( v8 )
    {
      sub_B96E90((__int64)v13, v8, 1);
      if ( v9 == v13 )
      {
        if ( v13[0] )
          sub_B91220((__int64)v13, v13[0]);
        goto LABEL_9;
      }
      v11 = *(_QWORD *)(v6 + 48);
      if ( !v11 )
      {
LABEL_15:
        v12 = (unsigned __int8 *)v13[0];
        *(_QWORD *)(v6 + 48) = v13[0];
        if ( v12 )
          sub_B976B0((__int64)v13, v12, v6 + 48);
        goto LABEL_9;
      }
    }
    else if ( v9 == v13 || (v11 = *(_QWORD *)(v6 + 48)) == 0 )
    {
LABEL_9:
      sub_BD6B90((unsigned __int8 *)v6, a4);
      return v6;
    }
    sub_B91220(v6 + 48, v11);
    goto LABEL_15;
  }
  return v6;
}
