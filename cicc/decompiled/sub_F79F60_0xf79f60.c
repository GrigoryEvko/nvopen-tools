// Function: sub_F79F60
// Address: 0xf79f60
//
__int64 __fastcall sub_F79F60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rsi
  char v7; // r15
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v11; // [rsp+8h] [rbp-68h]
  unsigned __int64 v12; // [rsp+30h] [rbp-40h]

  v3 = a2 - a1;
  v4 = a1;
  v5 = v3 >> 3;
  if ( v3 > 0 )
  {
    v6 = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
    v7 = *(_BYTE *)(v6 + 8);
    while ( 1 )
    {
      v8 = v5 >> 1;
      v9 = v4 + 8 * (v5 >> 1);
      if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v9 + 8LL) + 8LL) == 12 )
      {
        if ( v7 == 12 )
        {
          v11 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
          v12 = sub_BCAE30(v6);
          if ( v12 < sub_BCAE30(v11) )
            goto LABEL_8;
        }
LABEL_4:
        v5 >>= 1;
        if ( v8 <= 0 )
          return v4;
      }
      else
      {
        if ( v7 != 12 )
          goto LABEL_4;
LABEL_8:
        v4 = v9 + 8;
        v5 = v5 - v8 - 1;
        if ( v5 <= 0 )
          return v4;
      }
    }
  }
  return v4;
}
