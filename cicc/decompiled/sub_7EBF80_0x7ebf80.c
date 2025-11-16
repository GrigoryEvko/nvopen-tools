// Function: sub_7EBF80
// Address: 0x7ebf80
//
_BYTE *__fastcall sub_7EBF80(const __m128i *a1, __int64 a2, int a3, unsigned int a4)
{
  const __m128i *v6; // r12
  _QWORD *v8; // rax
  __int64 v9; // rdx
  __m128i *v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rax
  __int64 v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // r8
  __m128i *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  __m128i *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  __int64 v26; // [rsp+18h] [rbp-48h] BYREF
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  __int64 v28; // [rsp+28h] [rbp-38h] BYREF
  __int64 v29; // [rsp+30h] [rbp-30h] BYREF
  __m128i *v30[5]; // [rsp+38h] [rbp-28h] BYREF

  v6 = a1;
  if ( a1[1].m128i_i8[8] != 2 )
    goto LABEL_2;
  v14 = a1[3].m128i_i64[1];
  if ( *(_BYTE *)(v14 + 173) != 7 )
  {
    if ( (unsigned int)sub_7EBAB0(v14, v30) )
    {
      sub_7264E0((__int64)v6, 3);
      v17 = v30[0];
      v18 = v6->m128i_i64[0];
      v6[3].m128i_i64[1] = (__int64)v30[0]->m128i_i64;
      v19 = v17[7].m128i_i64[1];
      if ( v18 != v19 && !(unsigned int)sub_8D97D0(v18, v19, 0, v15, v16) )
        v6->m128i_i64[1] = v30[0][7].m128i_i64[1];
      goto LABEL_4;
    }
LABEL_2:
    if ( a3 )
      v6 = (const __m128i *)sub_7E8090(v6, a4);
LABEL_4:
    v8 = sub_726700(4);
    v9 = *(_QWORD *)(a2 + 120);
    v8[7] = a2;
    *v8 = v9;
    v6[1].m128i_i64[0] = (__int64)v8;
    v10 = sub_73D720(*(const __m128i **)(a2 + 120));
    v11 = sub_73DBF0(0x5Eu, (__int64)v10, (__int64)v6);
    v12 = sub_8D6540(*v11);
    return sub_73E130(v11, v12);
  }
  sub_7E13F0(v14, &v26, &v27, &v29, &v28);
  if ( qword_4F189F8 == a2 )
    return sub_7E0E90(v26, unk_4F06895);
  v20 = (__m128i *)sub_724DC0();
  v30[0] = v20;
  if ( v29 )
    sub_72D3B0(v29, (__int64)v20, 1);
  else
    sub_72BAF0((__int64)v20, v28, unk_4F06895);
  v21 = sub_7E1C50();
  sub_70FEE0((__int64)v30[0]->m128i_i64, v21, v22, v23, v24);
  v25 = sub_73A720(v30[0], v21);
  sub_724E30((__int64)v30);
  return v25;
}
