// Function: sub_386F3E0
// Address: 0x386f3e0
//
__int64 *__fastcall sub_386F3E0(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 *v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  __int64 v15; // [rsp+20h] [rbp-40h]
  bool v16; // [rsp+2Fh] [rbp-31h]

  v4 = a2 - (_QWORD)a1;
  v5 = v4 >> 4;
  v13 = a1;
  if ( v4 > 0 )
  {
    while ( 1 )
    {
      v6 = v5 >> 1;
      v7 = *a3;
      v8 = &v13[2 * (v5 >> 1)];
      v9 = v8[1];
      v15 = a3[1];
      v14 = *v8;
      v16 = *(_BYTE *)(sub_1456040(v15) + 8) == 15;
      if ( v16 == (*(_BYTE *)(sub_1456040(v9) + 8) == 15) )
      {
        if ( v14 == v7 )
        {
          if ( sub_1456260(v15) )
          {
            sub_1456260(v9);
            goto LABEL_8;
          }
          if ( !sub_1456260(v9) )
            goto LABEL_8;
        }
        else if ( v7 == sub_386EC30(v7, v14, a4) )
        {
          goto LABEL_8;
        }
LABEL_4:
        v5 >>= 1;
        if ( v6 <= 0 )
          return v13;
      }
      else
      {
        if ( *(_BYTE *)(sub_1456040(v15) + 8) == 15 )
          goto LABEL_4;
LABEL_8:
        v5 = v5 - v6 - 1;
        v13 = v8 + 2;
        if ( v5 <= 0 )
          return v13;
      }
    }
  }
  return v13;
}
