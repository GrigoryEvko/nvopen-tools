// Function: sub_EADE50
// Address: 0xeade50
//
__int64 __fastcall sub_EADE50(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned __int64 v5; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  unsigned __int8 *v10; // rsi
  char v11; // dl
  signed __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // r12
  __int64 v15; // r15
  __int64 v16; // r12
  __int64 v18; // [rsp+18h] [rbp-98h] BYREF
  __int64 v19[4]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v20; // [rsp+40h] [rbp-70h]
  _QWORD v21[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v22; // [rsp+70h] [rbp-40h]

  v5 = sub_ECD690(a1 + 40);
  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) || (unsigned __int8)sub_EAC8B0(a1, &v18) )
    return 1;
  if ( v18 < 0 )
  {
    v22 = 770;
    v19[0] = (__int64)"'";
    v19[2] = a2;
    v19[3] = a3;
    v20 = 1283;
    v21[0] = v19;
    v21[2] = "' directive with negative repeat count has no effect";
    sub_EA8060((_QWORD *)a1, v5, (__int64)v21, 0, 0);
    return 0;
  }
  v22 = 259;
  v21[0] = "expected comma";
  if ( (unsigned __int8)sub_ECE210(a1, 26, v21) )
    return 1;
  v8 = sub_ECD690(a1 + 40);
  v21[0] = 0;
  v9 = v8;
  if ( sub_EAC4D0(a1, v19, (__int64)v21) )
  {
    return 1;
  }
  else
  {
    v10 = (unsigned __int8 *)v19[0];
    if ( *(_BYTE *)v19[0] == 1 )
    {
      v11 = 8 * a4;
      v12 = *(_QWORD *)(v19[0] + 16);
      if ( v12 > 0xFFFFFFFFFFFFFFFFLL >> (64 - 8 * (unsigned __int8)a4)
        && (v12 < -(1LL << (v11 - 1)) || v12 > (1LL << (v11 - 1)) - 1) )
      {
        v21[0] = "literal value out of range for directive";
        v22 = 259;
        return (unsigned int)sub_ECDA70(a1, v9, v21, 0, 0);
      }
      v13 = v18;
      if ( v18 )
      {
        v14 = 0;
        do
        {
          ++v14;
          (*(void (__fastcall **)(_QWORD, signed __int64, _QWORD))(**(_QWORD **)(a1 + 232) + 536LL))(
            *(_QWORD *)(a1 + 232),
            v12,
            a4);
        }
        while ( v13 != v14 );
      }
    }
    else
    {
      v15 = v18;
      if ( v18 )
      {
        v16 = 0;
        while ( 1 )
        {
          ++v16;
          sub_E9A5B0(*(_QWORD *)(a1 + 232), v10);
          if ( v15 == v16 )
            break;
          v10 = (unsigned __int8 *)v19[0];
        }
      }
    }
    return (unsigned int)sub_ECE000(a1);
  }
}
