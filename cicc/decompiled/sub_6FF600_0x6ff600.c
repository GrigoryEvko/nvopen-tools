// Function: sub_6FF600
// Address: 0x6ff600
//
void __fastcall sub_6FF600(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 i; // rdx
  __int64 v10; // rax
  __int64 *v11; // rax
  __int64 v12; // rax
  bool v13; // zf
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-188h] BYREF
  _OWORD v21[9]; // [rsp+10h] [rbp-180h] BYREF
  __m128i v22; // [rsp+A0h] [rbp-F0h]
  __m128i v23; // [rsp+B0h] [rbp-E0h]
  __m128i v24; // [rsp+C0h] [rbp-D0h]
  __m128i v25; // [rsp+D0h] [rbp-C0h]
  __m128i v26; // [rsp+E0h] [rbp-B0h]
  __m128i v27; // [rsp+F0h] [rbp-A0h]
  __m128i v28; // [rsp+100h] [rbp-90h]
  __m128i v29; // [rsp+110h] [rbp-80h]
  __m128i v30; // [rsp+120h] [rbp-70h]
  __m128i v31; // [rsp+130h] [rbp-60h]
  __m128i v32; // [rsp+140h] [rbp-50h]
  __m128i v33; // [rsp+150h] [rbp-40h]
  __m128i v34; // [rsp+160h] [rbp-30h]

  v7 = a1[16];
  if ( !(_BYTE)v7 )
    goto LABEL_7;
  v8 = *(_QWORD *)a1;
  for ( i = *(unsigned __int8 *)(*(_QWORD *)a1 + 140LL); (_BYTE)i == 12; i = *(unsigned __int8 *)(v8 + 140) )
    v8 = *(_QWORD *)(v8 + 160);
  if ( (_BYTE)i )
  {
    if ( (a1[17] & 0xFD) == 1 )
    {
      sub_6FED50((__int64)a1, 1, a2, 0, 0, 0);
      return;
    }
    v21[0] = _mm_loadu_si128((const __m128i *)a1);
    v21[1] = _mm_loadu_si128((const __m128i *)a1 + 1);
    v21[2] = _mm_loadu_si128((const __m128i *)a1 + 2);
    v21[3] = _mm_loadu_si128((const __m128i *)a1 + 3);
    v21[4] = _mm_loadu_si128((const __m128i *)a1 + 4);
    v21[5] = _mm_loadu_si128((const __m128i *)a1 + 5);
    v21[6] = _mm_loadu_si128((const __m128i *)a1 + 6);
    v21[7] = _mm_loadu_si128((const __m128i *)a1 + 7);
    v21[8] = _mm_loadu_si128((const __m128i *)a1 + 8);
    switch ( (_BYTE)v7 )
    {
      case 2:
        v22 = _mm_loadu_si128((const __m128i *)a1 + 9);
        v23 = _mm_loadu_si128((const __m128i *)a1 + 10);
        v24 = _mm_loadu_si128((const __m128i *)a1 + 11);
        v25 = _mm_loadu_si128((const __m128i *)a1 + 12);
        v26 = _mm_loadu_si128((const __m128i *)a1 + 13);
        v27 = _mm_loadu_si128((const __m128i *)a1 + 14);
        v28 = _mm_loadu_si128((const __m128i *)a1 + 15);
        v29 = _mm_loadu_si128((const __m128i *)a1 + 16);
        v30 = _mm_loadu_si128((const __m128i *)a1 + 17);
        v31 = _mm_loadu_si128((const __m128i *)a1 + 18);
        v32 = _mm_loadu_si128((const __m128i *)a1 + 19);
        v33 = _mm_loadu_si128((const __m128i *)a1 + 20);
        v34 = _mm_loadu_si128((const __m128i *)a1 + 21);
        v12 = sub_724DC0(a1, a2, i, v7, a5, a6);
        v13 = a1[16] == 2;
        v20 = v12;
        v14 = v12;
        if ( v13 && a1[317] == 12 )
        {
          if ( a1[320] == 1 )
          {
            v17 = *((_QWORD *)a1 + 36);
            if ( !v17 )
              v17 = sub_72E9A0(a1 + 144);
          }
          else
          {
            v19 = sub_740630(a1 + 144);
            v17 = sub_73A720(v19);
          }
          v18 = sub_73E250(v17);
          sub_70FD90(v18, v20);
        }
        else
        {
          v15 = sub_740630(a1 + 144);
          sub_72D460(v15, v14);
        }
        v16 = v20;
        *(_QWORD *)(v20 + 128) = sub_72D600(*(_QWORD *)a1);
        sub_6E6A50(v16, (__int64)a1);
        sub_724E30(&v20);
        break;
      case 5:
        v22.m128i_i64[0] = *((_QWORD *)a1 + 18);
LABEL_11:
        sub_721090(a1);
      case 1:
        v22.m128i_i64[0] = *((_QWORD *)a1 + 18);
        v10 = sub_6F6F40((const __m128i *)a1, 0, i, v7, a5, a6);
        v11 = (__int64 *)sub_73E250(v10);
        *(__int64 *)((char *)v11 + 28) = *(_QWORD *)(a1 + 68);
        sub_6E70E0(v11, (__int64)a1);
        break;
      default:
        goto LABEL_11;
    }
    sub_6E4BC0((__int64)a1, (__int64)v21);
  }
  else
  {
LABEL_7:
    sub_6E6870((__int64)a1);
  }
}
