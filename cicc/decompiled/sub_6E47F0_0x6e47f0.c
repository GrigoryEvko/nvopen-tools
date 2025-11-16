// Function: sub_6E47F0
// Address: 0x6e47f0
//
__int64 __fastcall sub_6E47F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v5; // rbx
  __m128i *v6; // r14
  __int64 v7; // r12
  char v8; // al
  __m128i *v10; // r15
  unsigned int v11; // r9d
  __int64 v12; // rdx
  __int64 v13; // rdx
  char v14; // al
  __int64 v15; // rdx
  __int64 v16; // rdx
  const __m128i *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 i; // rax
  unsigned int v28; // eax
  __m128i *v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // [rsp+0h] [rbp-70h]
  __int64 v34; // [rsp+18h] [rbp-58h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  __int64 v36; // [rsp+18h] [rbp-58h]
  unsigned int v37; // [rsp+18h] [rbp-58h]
  unsigned int v38; // [rsp+2Ch] [rbp-44h] BYREF
  _QWORD v39[8]; // [rsp+30h] [rbp-40h] BYREF

  sub_7296C0(&v38);
  v4 = sub_726700(1);
  v5 = sub_726700(20);
  v6 = (__m128i *)sub_726700(2);
  v7 = sub_726700(1);
  sub_729730(v38);
  v8 = *(_BYTE *)(a2 + 16);
  if ( v8 == 2 )
  {
    v6[1].m128i_i8[9] &= 0xFCu;
    v6[3].m128i_i64[1] = a2 + 144;
    v19 = *(_QWORD *)a2;
    v6->m128i_i64[0] = *(_QWORD *)a2;
  }
  else
  {
    if ( v8 != 1 )
    {
      v10 = v6;
      v11 = 0;
      goto LABEL_4;
    }
    v18 = *(const __m128i **)(a2 + 144);
    *v6 = _mm_loadu_si128(v18);
    v19 = v6->m128i_i64[0];
    v6[1] = _mm_loadu_si128(v18 + 1);
    v6[2] = _mm_loadu_si128(v18 + 2);
    v6[3] = _mm_loadu_si128(v18 + 3);
    v6[4] = _mm_loadu_si128(v18 + 4);
    v20 = v18[5].m128i_i64[0];
    v6[1].m128i_i64[0] = 0;
    v6[5].m128i_i64[0] = v20;
  }
  v34 = a1;
  v10 = v6;
  v21 = sub_8D5CE0(v19, *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL));
  v22 = v34;
  if ( v21 )
  {
    if ( (v6[1].m128i_i8[9] & 3) != 0 )
    {
      v31 = v34;
      v35 = v21;
      v23 = sub_73E1B0(v6, v21);
      v24 = sub_73E4A0(v23, v35);
      v25 = sub_73DCD0(v24);
      v22 = v31;
    }
    else
    {
      v25 = sub_73E5A0(v6, v21);
      v22 = v34;
    }
    v10 = (__m128i *)v25;
  }
  *(_QWORD *)(v5 + 56) = v22;
  v26 = *(_QWORD *)(v22 + 152);
  *(_QWORD *)(v5 + 16) = v10;
  *(_QWORD *)v5 = v26;
  for ( i = *(_QWORD *)(v22 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v36 = *(_QWORD *)(i + 160);
  if ( (unsigned int)sub_8D32E0(v36) )
  {
    sub_73D8E0(v4, 106, v36, 1, v5);
    v28 = sub_8D32E0(a3);
    sub_73D8E0(v7, 4, a3, v28, v4);
  }
  else
  {
    sub_73D8E0(v4, 106, v36, 0, v5);
    sub_73D8E0(v7, 5, a3, 0, v4);
  }
  v39[0] = 0;
  v39[1] = 0;
  v37 = sub_7A30C0(v7, 1, 0, a4);
  sub_67E3D0(v39);
  v11 = v37;
LABEL_4:
  if ( (*(_BYTE *)(v4 - 8) & 1) != 0 )
  {
    *(_BYTE *)(v4 + 24) = 38;
    v12 = qword_4F06BB0;
    qword_4F06BB0 = v4;
    *(_QWORD *)(v4 + 80) = v12;
  }
  if ( (*(_BYTE *)(v5 - 8) & 1) != 0 )
  {
    *(_BYTE *)(v5 + 24) = 38;
    v13 = qword_4F06BB0;
    qword_4F06BB0 = v5;
    *(_QWORD *)(v5 + 80) = v13;
  }
  while ( 1 )
  {
    v14 = v10[-1].m128i_i8[8] & 1;
    if ( v10 == v6 || v10[1].m128i_i8[8] != 1 || v10[3].m128i_i8[8] != 14 )
      break;
    v29 = (__m128i *)v10[4].m128i_i64[1];
    if ( v14 )
    {
      v10[1].m128i_i8[8] = 38;
      v30 = qword_4F06BB0;
      qword_4F06BB0 = v10;
      v10[5].m128i_i64[0] = v30;
    }
    v10 = v29;
  }
  if ( v14 )
  {
    v10[1].m128i_i8[8] = 38;
    v15 = qword_4F06BB0;
    qword_4F06BB0 = v10;
    v10[5].m128i_i64[0] = v15;
  }
  if ( (*(_BYTE *)(v7 - 8) & 1) != 0 )
  {
    *(_BYTE *)(v7 + 24) = 38;
    v16 = qword_4F06BB0;
    qword_4F06BB0 = v7;
    *(_QWORD *)(v7 + 80) = v16;
  }
  return v11;
}
