// Function: sub_7F0F50
// Address: 0x7f0f50
//
void __fastcall sub_7F0F50(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __m128i *v3; // r12
  char v4; // al
  int v5; // ebx
  int v6; // eax
  __int64 v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // r15
  _QWORD *v10; // r14
  bool v11; // bl
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  const __m128i *v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  _BOOL8 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rcx
  _UNKNOWN *__ptr32 *v24; // r8
  __int64 v25; // rdi
  _QWORD *v26; // rax
  const __m128i *v27; // rdi
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-48h]
  const __m128i *v30; // [rsp+18h] [rbp-38h] BYREF

  v2 = 0;
  v3 = (__m128i *)a1;
  v4 = *(_BYTE *)(a1 + 24);
  if ( v4 == 10 )
  {
    v2 = a1;
    v4 = *(_BYTE *)(*(_QWORD *)(a1 + 56) + 24LL);
    v3 = *(__m128i **)(a1 + 56);
  }
  if ( v4 == 2 && sub_70FCE0(v3[3].m128i_i64[1]) )
  {
    v16 = (const __m128i *)sub_724DC0();
    v17 = v3[3].m128i_i64[1];
    v30 = v16;
    v21 = (unsigned int)sub_711520(v17, a2, v18, v19, v20) == 0;
    sub_72BAF0((__int64)v30, v21, 5u);
    v25 = v3[3].m128i_i64[1];
    if ( *(_QWORD *)(v25 + 144) )
    {
      if ( (unsigned int)sub_73A390(v25, v21, v22, v23, v24) )
        v26 = sub_730690(v3[3].m128i_i64[1]);
      else
        v26 = *(_QWORD **)(v3[3].m128i_i64[1] + 144);
      v27 = v30;
      v30[9].m128i_i64[0] = (__int64)v26;
    }
    else
    {
      v27 = v30;
    }
    v3[3].m128i_i64[1] = sub_73A460(v27, v21, v22, v23, v24);
    v3->m128i_i64[0] = v30[8].m128i_i64[0];
    sub_724E30((__int64)&v30);
  }
  else
  {
    v5 = unk_4F07520;
    v6 = sub_7E1F90(v3->m128i_i64[0]);
    v7 = v3->m128i_i64[0];
    if ( v6 || (v28 = sub_7E1F40(v7), v7 = v3->m128i_i64[0], v28) )
    {
      sub_8D2B50(v7);
    }
    else if ( !(v5 | (unsigned int)sub_8D2B50(v7)) )
    {
      v3[1].m128i_i8[10] |= 0x10u;
      goto LABEL_10;
    }
    if ( v3[1].m128i_i8[8] != 1 || !sub_730FB0(v3[3].m128i_i8[8]) )
    {
      v8 = (__int64 *)sub_730FF0(v3);
      v9 = sub_7E1E70(v8);
      v10 = sub_72BA30(5u);
      v29 = v3[1].m128i_i64[0];
      v11 = (v3[1].m128i_i8[9] & 4) != 0;
      sub_7266C0((__int64)v3, 1);
      v3[1].m128i_i64[0] = v29;
      v3[1].m128i_i8[9] = v3[1].m128i_i8[9] & 0xFB | (4 * v11);
      sub_73D8E0((__int64)v3, 0x3Bu, (__int64)v10, 0, (__int64)v9);
      v3[1].m128i_i8[11] |= 2u;
      sub_7F07E0((__int64)v3, 59, v12, v13, v14, v15);
      if ( v3[1].m128i_i8[8] != 1 || v3[3].m128i_i8[8] != 59 )
        sub_7F0F50(v3);
    }
  }
LABEL_10:
  if ( v2 )
    *(_QWORD *)v2 = **(_QWORD **)(v2 + 56);
}
