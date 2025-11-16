// Function: sub_22A84E0
// Address: 0x22a84e0
//
void __fastcall sub_22A84E0(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  __int64 v6; // r10
  char *v7; // r15
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  __int8 *v11; // r11
  __int64 v12; // r10
  char *v13; // r9
  __int64 v14; // r14
  __int64 v15; // rdx
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rsi
  __m128i v24; // xmm2
  __int64 v25; // rdi
  char *v27; // [rsp+10h] [rbp-50h]
  char *v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+20h] [rbp-40h]
  __int64 v32; // [rsp+28h] [rbp-38h]
  __int8 *v33; // [rsp+28h] [rbp-38h]
  __int64 v34; // [rsp+28h] [rbp-38h]
  __m128i *v35; // [rsp+28h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = a2;
      v8 = a4;
      if ( a5 + a4 == 2 )
      {
        v16 = (__int64)a2;
        v15 = a1;
LABEL_12:
        v31 = v15;
        v35 = (__m128i *)v16;
        if ( (unsigned __int8)sub_22A71D0(v16, v15) )
        {
          v18 = *(_QWORD *)(v31 + 48);
          *(_QWORD *)(v31 + 48) = v35[3].m128i_i64[0];
          v19 = _mm_loadu_si128(v35 + 1);
          v35[3].m128i_i64[0] = v18;
          v20 = *(_QWORD *)(v31 + 32);
          v21 = *(_QWORD *)(v31 + 40);
          v22 = *(_QWORD *)(v31 + 16);
          v23 = *(_QWORD *)(v31 + 24);
          *(__m128i *)(v31 + 16) = v19;
          *(__m128i *)(v31 + 32) = _mm_loadu_si128(v35 + 2);
          v24 = _mm_loadu_si128(v35);
          v35[1].m128i_i64[0] = v22;
          v35[1].m128i_i64[1] = v23;
          v35[2].m128i_i64[0] = v20;
          v35[2].m128i_i64[1] = v21;
          v25 = *(_QWORD *)v31;
          LOBYTE(v20) = *(_BYTE *)(v31 + 10);
          LOWORD(v23) = *(_WORD *)(v31 + 8);
          LODWORD(v21) = *(_DWORD *)(v31 + 12);
          *(__m128i *)v31 = v24;
          v35->m128i_i64[0] = v25;
          v35->m128i_i16[4] = v23;
          v35->m128i_i8[10] = v20;
          v35->m128i_i32[3] = v21;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v29 = v6;
        v9 = v8 / 2;
        v32 = v6 + 56 * (v8 / 2);
        v10 = sub_22A7890((__int64)v7, a3, v32);
        v11 = (__int8 *)v32;
        v12 = v29;
        v13 = (char *)v10;
        v14 = 0x6DB6DB6DB6DB6DB7LL * ((v10 - (__int64)v7) >> 3);
        while ( 1 )
        {
          v30 = v12;
          v27 = v13;
          v5 -= v14;
          v33 = v11;
          v28 = sub_22A6590(v11, v7, v13);
          sub_22A84E0(v30, v33, v28, v9, v14);
          v8 -= v9;
          if ( !v8 )
            break;
          v15 = (__int64)v28;
          v16 = (__int64)v27;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v7 = v27;
          v6 = (__int64)v28;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v34 = v6;
          v14 = v5 / 2;
          v17 = sub_22A79B0(v6, (__int64)v7, (__int64)&v7[56 * (v5 / 2)]);
          v12 = v34;
          v13 = &v7[56 * (v5 / 2)];
          v11 = (__int8 *)v17;
          v9 = 0x6DB6DB6DB6DB6DB7LL * ((v17 - v34) >> 3);
        }
      }
    }
  }
}
