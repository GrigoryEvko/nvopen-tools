// Function: sub_1E78B90
// Address: 0x1e78b90
//
__int64 *__fastcall sub_1E78B90(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 v5; // r15
  __int64 v6; // rdi
  unsigned __int64 v7; // rax
  __int64 v8; // rbx
  __int64 *v9; // r12
  __int64 v10; // rdi
  __int64 v11; // r14
  __int64 v12; // r13
  unsigned __int64 v15; // [rsp+10h] [rbp-50h]
  __int64 v16; // [rsp+18h] [rbp-48h]
  __int64 *v17; // [rsp+20h] [rbp-40h]
  __int64 v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v4 = a2 - (_QWORD)a1;
  v5 = v4 >> 3;
  v17 = a1;
  v18[0] = a4;
  if ( v4 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v8 = v5 >> 1;
        v9 = &v17[v5 >> 1];
        v10 = *(_QWORD *)(v18[0] + 280);
        v16 = v18[0];
        v11 = *v9;
        v12 = *a3;
        if ( v10 )
        {
          v15 = sub_1DDC3C0(v10, *v9);
          v6 = *(_QWORD *)(v16 + 280);
          if ( v6 )
          {
            v7 = sub_1DDC3C0(v6, v12);
            if ( v15 )
            {
              if ( v7 )
                break;
            }
          }
        }
        if ( sub_1E78020((__int64)v18, v11, v12) )
          goto LABEL_7;
LABEL_10:
        v5 >>= 1;
        if ( v8 <= 0 )
          return v17;
      }
      if ( v15 >= v7 )
        goto LABEL_10;
LABEL_7:
      v17 = v9 + 1;
      v5 = v5 - v8 - 1;
    }
    while ( v5 > 0 );
  }
  return v17;
}
