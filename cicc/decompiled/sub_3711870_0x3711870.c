// Function: sub_3711870
// Address: 0x3711870
//
__int64 *__fastcall sub_3711870(__int64 *a1, _QWORD *a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v4; // r13
  unsigned __int64 v6; // r15
  __int64 v7; // r10
  __int64 v8; // r10
  unsigned __int64 v9; // rbx
  unsigned __int8 v10; // al
  unsigned __int64 v11; // rax
  unsigned __int64 v13; // rdx
  unsigned __int16 v14; // r15
  _BYTE *v15; // rsi
  _BYTE *v16; // rsi
  unsigned __int16 v17; // cx
  __int64 v18; // [rsp+10h] [rbp-80h]
  __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+10h] [rbp-80h]
  unsigned __int8 v21; // [rsp+25h] [rbp-6Bh] BYREF
  unsigned __int16 v22; // [rsp+26h] [rbp-6Ah] BYREF
  unsigned __int64 v23; // [rsp+28h] [rbp-68h] BYREF
  __m128i v24[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v25; // [rsp+50h] [rbp-40h]

  v4 = a2 + 2;
  if ( a2[7] && !a2[9] && !a2[8] )
  {
    v25 = 257;
    sub_370BC10(&v23, a2 + 2, &v22, v24);
    v13 = v23 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
LABEL_24:
      v23 = v13 | 1;
      *a1 = 0;
      sub_9C6670(a1, &v23);
      sub_9C66B0((__int64 *)&v23);
      return a1;
    }
    v23 = 0;
    sub_9C66B0((__int64 *)&v23);
    if ( v22 )
    {
      v14 = 0;
      v20 = (__int64)(a4 + 3);
      do
      {
        v25 = 257;
        sub_3702900(&v23, v4, (char *)&v21, v24);
        v13 = v23 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_24;
        v23 = 0;
        sub_9C66B0((__int64 *)&v23);
        v15 = (_BYTE *)a4[4];
        v24[0].m128i_i8[0] = v21 & 0xF;
        if ( v15 == (_BYTE *)a4[5] )
        {
          sub_3711700(v20, v15, v24);
          v16 = (_BYTE *)a4[4];
        }
        else
        {
          if ( v15 )
          {
            *v15 = v21 & 0xF;
            v15 = (_BYTE *)a4[4];
          }
          v16 = v15 + 1;
          a4[4] = v16;
        }
        v17 = v22;
        if ( v14 + 1 < v22 )
        {
          v24[0].m128i_i8[0] = v21 >> 4;
          if ( (_BYTE *)a4[5] == v16 )
          {
            sub_3711700(v20, v16, v24);
            v17 = v22;
          }
          else
          {
            if ( v16 )
            {
              *v16 = v21 >> 4;
              v16 = (_BYTE *)a4[4];
              v17 = v22;
            }
            a4[4] = v16 + 1;
          }
        }
        v14 += 2;
      }
      while ( v14 < v17 );
    }
LABEL_26:
    v24[0].m128i_i64[0] = 0;
    *a1 = 1;
    sub_9C66B0(v24[0].m128i_i64);
    return a1;
  }
  v6 = a4[2];
  if ( v6 )
  {
    v7 = a4[1];
  }
  else
  {
    v7 = a4[3];
    v6 = a4[4] - v7;
  }
  v18 = v7;
  v22 = v6;
  v24[0].m128i_i64[0] = (__int64)"VFEntryCount";
  v25 = 259;
  sub_370BC10(&v23, a2 + 2, &v22, v24);
  v8 = v18;
  v9 = v23 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v9 | 1;
    return a1;
  }
  if ( !v6 )
    goto LABEL_26;
  while ( 1 )
  {
    v10 = 16 * *(_BYTE *)(v8 + v9);
    v21 = v10;
    if ( v6 > v9 + 1 )
      v21 = *(_BYTE *)(v8 + v9 + 1) | v10;
    v25 = 257;
    v19 = v8;
    sub_3702900(&v23, v4, (char *)&v21, v24);
    v8 = v19;
    if ( (v23 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      break;
    v9 += 2LL;
    if ( v6 <= v9 )
      goto LABEL_26;
  }
  v11 = v23 & 0xFFFFFFFFFFFFFFFELL | 1;
  v23 = 0;
  *a1 = v11;
  sub_9C66B0((__int64 *)&v23);
  return a1;
}
