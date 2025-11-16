// Function: sub_10C6580
// Address: 0x10c6580
//
_QWORD *__fastcall sub_10C6580(__int64 a1, unsigned __int64 a2, unsigned __int8 a3, unsigned __int8 a4)
{
  __int64 v6; // rsi
  _QWORD *v7; // rdi
  __int64 *v8; // rdi
  __int64 *v9; // rbx
  __int64 v10; // r15
  __int64 v11; // rsi
  __int64 *v12; // r13
  __int64 v13; // rsi
  unsigned __int8 *v14; // rsi
  _QWORD *v16; // [rsp+8h] [rbp-88h]
  __int64 *v17; // [rsp+10h] [rbp-80h]
  unsigned __int8 *v19; // [rsp+28h] [rbp-68h] BYREF
  __int64 *v20; // [rsp+30h] [rbp-60h] BYREF
  __int64 v21; // [rsp+38h] [rbp-58h]
  _BYTE v22[80]; // [rsp+40h] [rbp-50h] BYREF

  v6 = a3;
  v20 = (__int64 *)v22;
  v21 = 0x400000000LL;
  if ( (unsigned __int8)sub_F5BB40(a2, a3, a4, (__int64)&v20) )
  {
    v7 = (_QWORD *)v20[(unsigned int)v21 - 1];
    v16 = v7;
    LODWORD(v21) = v21 - 1;
    sub_B43D10(v7);
    v8 = v20;
    v17 = &v20[(unsigned int)v21];
    if ( v17 == v20 )
      goto LABEL_17;
    v9 = v20;
    while ( 1 )
    {
      v10 = *v9;
      v11 = *(_QWORD *)(a2 + 48);
      v12 = (__int64 *)(*v9 + 48);
      v19 = (unsigned __int8 *)v11;
      if ( v11 )
        break;
      if ( v12 != (__int64 *)&v19 )
      {
        v13 = *(_QWORD *)(v10 + 48);
        if ( v13 )
          goto LABEL_11;
      }
LABEL_7:
      v6 = v10;
      ++v9;
      sub_F15FC0(*(_QWORD *)(a1 + 40), v10);
      if ( v17 == v9 )
      {
        v8 = v20;
        goto LABEL_17;
      }
    }
    sub_B96E90((__int64)&v19, v11, 1);
    if ( v12 == (__int64 *)&v19 )
    {
      if ( v19 )
        sub_B91220((__int64)&v19, (__int64)v19);
      goto LABEL_7;
    }
    v13 = *(_QWORD *)(v10 + 48);
    if ( v13 )
LABEL_11:
      sub_B91220((__int64)v12, v13);
    v14 = v19;
    *(_QWORD *)(v10 + 48) = v19;
    if ( v14 )
      sub_B976B0((__int64)&v19, v14, (__int64)v12);
    goto LABEL_7;
  }
  v16 = 0;
  v8 = v20;
LABEL_17:
  if ( v8 != (__int64 *)v22 )
    _libc_free(v8, v6);
  return v16;
}
