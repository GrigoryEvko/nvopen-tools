// Function: sub_2A47640
// Address: 0x2a47640
//
void __fastcall sub_2A47640(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  signed __int64 v6; // r14
  __int64 v7; // r11
  __int64 v8; // r10
  signed __int64 v9; // r13
  __int64 v10; // rcx
  __m128i *v11; // r12
  __int64 v12; // rax
  const __m128i *v13; // r10
  __int64 v14; // r11
  __m128i *v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rdx
  __int64 v18; // rax
  char v19; // al
  __int32 v20; // r8d
  __int64 v21; // r9
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int8 v25; // al
  __int64 v27; // [rsp+10h] [rbp-50h]
  __m128i *v28; // [rsp+10h] [rbp-50h]
  const __m128i *v29; // [rsp+18h] [rbp-48h]
  __int64 v30; // [rsp+18h] [rbp-48h]
  const __m128i *v31; // [rsp+18h] [rbp-48h]
  __int64 v32; // [rsp+20h] [rbp-40h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  __int64 v34; // [rsp+20h] [rbp-40h]
  __int64 v35[7]; // [rsp+28h] [rbp-38h] BYREF

  v35[0] = a6;
  if ( a4 )
  {
    v6 = a5;
    if ( a5 )
    {
      v7 = a1;
      v8 = (__int64)a2;
      v9 = a4;
      if ( a5 + a4 == 2 )
      {
        v15 = a2;
        v17 = a1;
LABEL_12:
        v34 = v17;
        sub_2A44DC0((__int64)v35, (__int64)v15, v17);
        if ( v19 )
        {
          v20 = *(_DWORD *)(v34 + 8);
          v21 = *(_QWORD *)v34;
          v22 = *(_QWORD *)(v34 + 16);
          *(__m128i *)v34 = _mm_loadu_si128(v15);
          v23 = *(_QWORD *)(v34 + 24);
          v24 = *(_QWORD *)(v34 + 32);
          v25 = *(_BYTE *)(v34 + 40);
          *(__m128i *)(v34 + 16) = _mm_loadu_si128(v15 + 1);
          *(__m128i *)(v34 + 32) = _mm_loadu_si128(v15 + 2);
          v15->m128i_i64[0] = v21;
          v15->m128i_i32[2] = v20;
          v15[1].m128i_i64[0] = v22;
          v15[1].m128i_i64[1] = v23;
          v15[2].m128i_i64[0] = v24;
          v15[2].m128i_i8[8] = v25;
        }
      }
      else
      {
        v10 = v35[0];
        if ( v9 <= a5 )
          goto LABEL_10;
LABEL_5:
        v27 = v7;
        v29 = (const __m128i *)v8;
        v32 = v9 / 2;
        v11 = (__m128i *)(v7 + 16 * (v9 / 2 + ((v9 + ((unsigned __int64)v9 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v12 = sub_2A47500(v8, a3, (__int64)v11, v10);
        v13 = v29;
        v14 = v27;
        v15 = (__m128i *)v12;
        v16 = 0xAAAAAAAAAAAAAAABLL * ((v12 - (__int64)v29) >> 4);
        while ( 1 )
        {
          v30 = v14;
          v28 = sub_2A445A0(v11, v13, v15);
          v6 -= v16;
          sub_2A47640(v30, v11, v28, v32, v16, v35[0]);
          v9 -= v32;
          if ( !v9 )
            break;
          v17 = (__int64)v28;
          if ( !v6 )
            break;
          if ( v6 + v9 == 2 )
            goto LABEL_12;
          v10 = v35[0];
          v8 = (__int64)v15;
          v7 = (__int64)v28;
          if ( v9 > v6 )
            goto LABEL_5;
LABEL_10:
          v31 = (const __m128i *)v8;
          v33 = v7;
          v16 = v6 / 2;
          v15 = (__m128i *)(v8 + 16 * (v6 / 2 + ((v6 + ((unsigned __int64)v6 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v18 = sub_2A475A0(v7, v8, (__int64)v15, v10);
          v14 = v33;
          v13 = v31;
          v11 = (__m128i *)v18;
          v32 = 0xAAAAAAAAAAAAAAABLL * ((v18 - v33) >> 4);
        }
      }
    }
  }
}
