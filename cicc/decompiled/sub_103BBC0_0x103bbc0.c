// Function: sub_103BBC0
// Address: 0x103bbc0
//
__int64 __fastcall sub_103BBC0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r10
  __int64 v7; // r11
  __int64 v8; // r13
  __int64 v9; // r14
  unsigned __int8 *v13; // rdx
  int v14; // esi
  __int64 v16; // rdi
  _QWORD *v17; // [rsp+8h] [rbp-78h]
  __m128i v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+20h] [rbp-60h]
  __int64 v20; // [rsp+28h] [rbp-58h]
  __int64 v21; // [rsp+30h] [rbp-50h]
  __int64 v22; // [rsp+38h] [rbp-48h]

  v13 = *(unsigned __int8 **)(a2 + 72);
  v14 = *v13;
  if ( (unsigned __int8)(v14 - 34) > 0x33u )
  {
LABEL_2:
    v17 = a3;
    sub_D66840(&v18, v13);
    v8 = v18.m128i_i64[1];
    v9 = v18.m128i_i64[0];
    v7 = v19;
    v6 = v20;
    a6 = v21;
    a5 = v22;
    v13 = *(unsigned __int8 **)(a2 + 72);
    a3 = v17;
LABEL_3:
    v18.m128i_i64[0] = v9;
    v18.m128i_i64[1] = v8;
    v19 = v7;
    v20 = v6;
    v21 = a6;
    v22 = a5;
    return sub_103B8F0(a1, &v18, v13, a3);
  }
  v16 = 0x8000000000041LL;
  if ( !_bittest64(&v16, (unsigned int)(v14 - 34)) )
  {
    if ( (_BYTE)v14 == 64 )
      goto LABEL_3;
    goto LABEL_2;
  }
  v18.m128i_i64[0] = 0;
  v18.m128i_i64[1] = -1;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  return sub_103B8F0(a1, &v18, v13, a3);
}
