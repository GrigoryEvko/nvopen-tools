// Function: sub_BFABA0
// Address: 0xbfaba0
//
void __fastcall sub_BFABA0(__int64 *a1, __int64 a2)
{
  unsigned __int16 v3; // ax
  __int64 v4; // r13
  unsigned __int16 v5; // bx
  unsigned __int64 v6; // rdx
  int v7; // edi
  __int64 v8; // r13
  _BYTE *v9; // rax
  __int64 v10; // rax
  unsigned __int8 v11; // cl
  const char *v12; // rax
  __int64 v13; // rax
  char *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r8
  _BYTE *v19; // rax
  __m128i v20; // [rsp+0h] [rbp-C0h] BYREF
  char *v21; // [rsp+10h] [rbp-B0h]
  __int64 v22; // [rsp+18h] [rbp-A8h]
  __int16 v23; // [rsp+20h] [rbp-A0h]
  __m128i v24[2]; // [rsp+30h] [rbp-90h] BYREF
  char v25; // [rsp+50h] [rbp-70h]
  char v26; // [rsp+51h] [rbp-6Fh]
  __m128i v27[2]; // [rsp+60h] [rbp-60h] BYREF
  char v28; // [rsp+80h] [rbp-40h]
  char v29; // [rsp+81h] [rbp-3Fh]

  if ( ((*(_WORD *)(a2 + 2) >> 1) & 7) != 1 )
  {
    v3 = (*(_WORD *)(a2 + 2) >> 4) & 0x1F;
    v4 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
    v5 = v3;
    v6 = *(unsigned __int8 *)(v4 + 8);
    if ( v3 )
    {
      v7 = v3;
      if ( (unsigned int)v3 - 11 <= 3 )
      {
        v11 = *(_BYTE *)(v4 + 8);
        if ( (unsigned int)(unsigned __int8)v6 - 17 <= 1 )
          v11 = *(_BYTE *)(**(_QWORD **)(v4 + 16) + 8LL);
        if ( (v11 <= 3u || v11 == 5 || (v11 & 0xFD) == 4) && (_BYTE)v6 != 18 )
        {
LABEL_5:
          sub_BDBDF0((__int64)a1, v4, (_BYTE *)a2);
          if ( v5 > 0x12u )
          {
            v8 = *a1;
            v29 = 1;
            v27[0].m128i_i64[0] = (__int64)"Invalid binary operation!";
            v28 = 3;
            if ( !v8 )
            {
              *((_BYTE *)a1 + 152) = 1;
              return;
            }
            sub_CA0E80(v27, v8);
            v9 = *(_BYTE **)(v8 + 32);
            if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
            {
              sub_CB5D20(v8, 10);
            }
            else
            {
              *(_QWORD *)(v8 + 32) = v9 + 1;
              *v9 = 10;
            }
            v10 = *a1;
            *((_BYTE *)a1 + 152) = 1;
            if ( v10 )
              goto LABEL_10;
            return;
          }
LABEL_16:
          sub_BF6FE0((__int64)a1, a2);
          return;
        }
        v26 = 1;
        v12 = " operand must have floating-point or fixed vector of floating-point type!";
      }
      else
      {
        if ( (_BYTE)v6 == 12 )
          goto LABEL_5;
        v26 = 1;
        v12 = " operand must have integer type!";
      }
      v24[0].m128i_i64[0] = (__int64)v12;
      v25 = 3;
    }
    else
    {
      if ( (unsigned __int8)v6 <= 0xCu && (v13 = 4143, _bittest64(&v13, v6)) || (v6 & 0xFD) == 4 || (_BYTE)v6 == 14 )
      {
        sub_BDBDF0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL), (_BYTE *)a2);
        goto LABEL_16;
      }
      v26 = 1;
      v7 = 0;
      v24[0].m128i_i64[0] = (__int64)" operand must have integer or floating point type!";
      v25 = 3;
    }
    v14 = sub_B4D7D0(v7);
    v22 = v15;
    v23 = 1283;
    v21 = v14;
    v20.m128i_i64[0] = (__int64)"atomicrmw ";
    sub_9C6370(v27, &v20, v24, 1283, v16, v17);
    sub_BDBF70(a1, (__int64)v27);
    if ( *a1 )
    {
      sub_BDBD80((__int64)a1, (_BYTE *)a2);
      v18 = *a1;
      v19 = *(_BYTE **)(*a1 + 32);
      if ( (unsigned __int64)v19 >= *(_QWORD *)(*a1 + 24) )
      {
        v18 = sub_CB5D20(*a1, 32);
      }
      else
      {
        *(_QWORD *)(v18 + 32) = v19 + 1;
        *v19 = 32;
      }
      sub_A587F0(v4, v18, 0, 0);
    }
    return;
  }
  v29 = 1;
  v27[0].m128i_i64[0] = (__int64)"atomicrmw instructions cannot be unordered.";
  v28 = 3;
  sub_BDBF70(a1, (__int64)v27);
  if ( *a1 )
LABEL_10:
    sub_BDBD80((__int64)a1, (_BYTE *)a2);
}
