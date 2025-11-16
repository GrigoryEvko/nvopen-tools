// Function: sub_28FEA70
// Address: 0x28fea70
//
unsigned __int8 *__fastcall sub_28FEA70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _BYTE *v6; // rbx
  unsigned __int8 *v7; // r14
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // r15
  const char *v11; // rdx
  const char *v18[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v19; // [rsp+50h] [rbp-40h]

  if ( a1 + 8 * a2 == a1 )
    return 0;
  v6 = 0;
  v7 = 0;
  v8 = a1 + 8 * a2;
  do
  {
    while ( 1 )
    {
      v9 = (__int64)v6;
      v6 = *(_BYTE **)(v8 - 8);
      v10 = (__int64)v7;
      v7 = (unsigned __int8 *)sub_B47F80(v6);
      sub_B44220(v7, a3, a4);
      v18[0] = sub_BD5D20((__int64)v6);
      v19 = 773;
      v18[1] = v11;
      v18[2] = ".remat";
      sub_BD6B50(v7, v18);
      if ( !v10 )
        break;
      sub_BD2ED0((__int64)v7, v9, v10);
LABEL_4:
      v8 -= 8;
      if ( a1 == v8 )
        return v7;
    }
    if ( a5 == a6 )
      goto LABEL_4;
    v8 -= 8;
    sub_BD2ED0((__int64)v7, a5, a6);
  }
  while ( a1 != v8 );
  return v7;
}
