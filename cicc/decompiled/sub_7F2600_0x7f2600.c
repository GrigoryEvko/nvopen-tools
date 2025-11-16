// Function: sub_7F2600
// Address: 0x7f2600
//
char __fastcall sub_7F2600(__int64 a1, __m128i *a2)
{
  __int64 v2; // r15
  __m128i *v3; // r13
  _QWORD *v4; // r12
  bool v5; // zf
  char result; // al
  __int64 v7; // rdx
  const __m128i *v8; // r14
  void *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // [rsp+8h] [rbp-E8h]
  __int64 v14; // [rsp+10h] [rbp-E0h]
  __int64 v15; // [rsp+18h] [rbp-D8h]
  __int64 v16; // [rsp+18h] [rbp-D8h]
  int v17; // [rsp+20h] [rbp-D0h] BYREF
  __m128i *v18; // [rsp+28h] [rbp-C8h]
  int v19; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v20; // [rsp+48h] [rbp-A8h]
  char v21[144]; // [rsp+60h] [rbp-90h] BYREF

  v2 = 0;
  v3 = (__m128i *)a1;
  v4 = (_QWORD *)a1;
  if ( *(_BYTE *)(a1 + 24) == 10 )
  {
    v2 = *(_QWORD *)(a1 + 64);
    v3 = *(__m128i **)(a1 + 56);
    sub_7E18E0((__int64)v21, 0, v2);
    sub_7E1790((__int64)&v19);
    sub_7E9190(v2, (__int64)&v19);
    sub_7E0190(a1);
  }
  if ( (*(_BYTE *)(a1 + 25) & 1) != 0 )
  {
    if ( (v3[1].m128i_i8[9] & 4) == 0 )
      goto LABEL_5;
    sub_7E0A10(v3);
    if ( *(_BYTE *)(a1 + 24) == 10 )
      *(_BYTE *)(a1 + 25) &= ~1u;
  }
  if ( (v3[1].m128i_i8[9] & 4) != 0
    && v3[1].m128i_i8[8] == 1
    && v3[3].m128i_i8[8] == 5
    && (unsigned int)sub_8D2600(v3->m128i_i64[0]) )
  {
    sub_730620((__int64)v3, (const __m128i *)v3[4].m128i_i64[1]);
  }
LABEL_5:
  if ( !dword_4D04380
    || !a2
    || v3 != (__m128i *)a1
    || v3[1].m128i_i8[8] != 1
    || (unsigned __int8)(v3[3].m128i_i8[8] - 105) > 4u )
  {
    sub_7EE560(v3, 0);
    if ( v2 )
      goto LABEL_10;
    return (unsigned __int8)sub_7E7010(v4);
  }
  sub_7E0190((__int64)v3);
  result = sub_7F1E10(v3, 0, a2, &v17);
  if ( !v17 )
  {
    if ( v2 )
    {
LABEL_10:
      if ( !(unsigned int)sub_7E71E0(v2, 0, 1) )
      {
LABEL_11:
        v5 = v19 == 5;
        *v4 = v3->m128i_i64[0];
        if ( !v5 )
        {
          sub_7E1780((__int64)v3, (__int64)&v17);
          sub_7E25D0(v20, &v17);
        }
        sub_7E1AA0();
        if ( !dword_4D03F8C )
        {
          sub_733650(v4[8]);
          sub_730620((__int64)v4, v3);
        }
        return (unsigned __int8)sub_7E7010(v4);
      }
      v17 = 4;
      v18 = v3;
      if ( (v3[1].m128i_i8[9] & 4) != 0 || (v15 = v3->m128i_i64[0], (unsigned int)sub_8D2600(v3->m128i_i64[0])) )
      {
LABEL_25:
        sub_7E7530(v2, (__int64)&v17);
        goto LABEL_11;
      }
      v7 = v15;
      v8 = v3;
      if ( v3[1].m128i_i8[8] != 1 || v3[3].m128i_i8[8] != 59 || v3[3].m128i_i8[9] != 2 )
      {
LABEL_32:
        v13 = v7;
        v16 = (__int64)sub_7E7CA0(v7);
        v9 = sub_730FF0(v8);
        v10 = sub_7E2BE0(v16, (__int64)v9);
        *(_QWORD *)(v10 + 16) = sub_73E830(v16);
        v14 = v8[1].m128i_i64[0];
        LOBYTE(v16) = (v8[1].m128i_i8[9] & 4) != 0;
        sub_7266C0((__int64)v8, 1);
        v8[1].m128i_i64[0] = v14;
        v8[1].m128i_i8[9] = v8[1].m128i_i8[9] & 0xFB | (4 * v16);
        sub_73D8E0((__int64)v8, 0x5Bu, v13, 0, v10);
        v18 = (__m128i *)v10;
        goto LABEL_25;
      }
      v11 = v3[4].m128i_i64[1];
      v8 = *(const __m128i **)(v11 + 16);
      if ( *(_BYTE *)(v11 + 24) == 2 )
      {
        v12 = *(_QWORD *)(v11 + 56);
      }
      else
      {
        if ( v8[1].m128i_i8[8] != 2 )
        {
LABEL_39:
          v8 = v3;
          goto LABEL_32;
        }
        v12 = v8[3].m128i_i64[1];
        v8 = (const __m128i *)v3[4].m128i_i64[1];
      }
      if ( v12 )
      {
        v7 = v8->m128i_i64[0];
        goto LABEL_32;
      }
      goto LABEL_39;
    }
    return (unsigned __int8)sub_7E7010(v4);
  }
  if ( v2 )
  {
    v3 = 0;
    v4 = 0;
    goto LABEL_10;
  }
  return result;
}
