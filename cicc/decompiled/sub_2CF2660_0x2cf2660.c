// Function: sub_2CF2660
// Address: 0x2cf2660
//
__int64 __fastcall sub_2CF2660(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r14
  _QWORD *v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rsi
  __int64 v11; // [rsp+8h] [rbp-68h]
  unsigned __int8 v12; // [rsp+17h] [rbp-59h]
  __int64 v13; // [rsp+18h] [rbp-58h]
  __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  __int64 v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h]
  unsigned int v17; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 80);
  v11 = a2 + 72;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v13 = v2;
  if ( v2 == a2 + 72 )
  {
    v12 = 0;
    v8 = 0;
    v9 = 0;
  }
  else
  {
    v12 = 0;
    do
    {
      v3 = *(_QWORD *)(v13 + 32);
      v4 = v13 + 24;
      v13 = *(_QWORD *)(v13 + 8);
      while ( v4 != v3 )
      {
        v5 = v3;
        v3 = *(_QWORD *)(v3 + 8);
        if ( *(_BYTE *)(v5 - 24) == 79 )
        {
          v6 = v5 - 24;
          v7 = sub_2CF0090(a1, v5 - 24, (__int64)&v14);
          if ( v7 )
          {
            sub_BD84D0(v6, (__int64)v7);
            v12 = 1;
          }
        }
      }
    }
    while ( v11 != v13 );
    v8 = v15;
    v9 = 16LL * v17;
  }
  sub_C7D6A0(v8, v9, 8);
  return v12;
}
