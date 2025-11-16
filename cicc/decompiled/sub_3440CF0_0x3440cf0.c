// Function: sub_3440CF0
// Address: 0x3440cf0
//
void __fastcall sub_3440CF0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5)
{
  signed __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // r11
  signed __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rax
  const __m128i *v11; // r11
  const __m128i *v12; // r10
  const __m128i *v13; // r9
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  __m128i *v16; // r9
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned int v20; // ecx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __m128i *v24; // [rsp+8h] [rbp-48h]
  unsigned __int64 v25; // [rsp+10h] [rbp-40h]
  const __m128i *v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]

  if ( a4 )
  {
    v5 = a5;
    if ( a5 )
    {
      v6 = a1;
      v7 = (__int64)a2;
      v8 = a4;
      if ( a4 + a5 == 2 )
      {
        v16 = a2;
        v15 = a1;
LABEL_12:
        v18 = v16[1].m128i_u32[0];
        v19 = *(unsigned int *)(v15 + 16);
        if ( (unsigned int)v18 > 6 || (v20 = dword_44E2140[v18], (unsigned int)v19 > 6) )
          BUG();
        if ( v20 > dword_44E2140[v19] )
        {
          v21 = *(_QWORD *)v15;
          v22 = *(_QWORD *)(v15 + 8);
          *(__m128i *)v15 = _mm_loadu_si128(v16);
          v16->m128i_i64[0] = v21;
          LODWORD(v21) = v16[1].m128i_i32[0];
          v16->m128i_i64[1] = v22;
          LODWORD(v22) = *(_DWORD *)(v15 + 16);
          *(_DWORD *)(v15 + 16) = v21;
          v16[1].m128i_i32[0] = v22;
        }
      }
      else
      {
        if ( a4 <= a5 )
          goto LABEL_10;
LABEL_5:
        v9 = v8 / 2;
        v10 = sub_3440630(v7, a3, v6 + 8 * (v8 / 2 + ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v12 = (const __m128i *)(v6 + 8 * (v8 / 2 + ((v8 + ((unsigned __int64)v8 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v13 = (const __m128i *)v10;
        v14 = 0xAAAAAAAAAAAAAAABLL * ((v10 - (__int64)v11) >> 3);
        while ( 1 )
        {
          v24 = (__m128i *)v13;
          v26 = v12;
          v5 -= v14;
          v25 = sub_343F700(v12, v11, v13);
          sub_3440CF0(v6, v26, v25, v9, v14);
          v8 -= v9;
          if ( !v8 )
            break;
          v15 = v25;
          v16 = v24;
          if ( !v5 )
            break;
          if ( v5 + v8 == 2 )
            goto LABEL_12;
          v7 = (__int64)v24;
          v6 = v25;
          if ( v8 > v5 )
            goto LABEL_5;
LABEL_10:
          v14 = v5 / 2;
          v27 = v7 + 8 * (v5 / 2 + ((v5 + ((unsigned __int64)v5 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
          v17 = sub_34406C0(v6, v7, v27);
          v13 = (const __m128i *)v27;
          v12 = (const __m128i *)v17;
          v9 = 0xAAAAAAAAAAAAAAABLL * ((v17 - v6) >> 3);
        }
      }
    }
  }
}
