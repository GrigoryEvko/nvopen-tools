// Function: sub_6FF9F0
// Address: 0x6ff9f0
//
__int64 __fastcall sub_6FF9F0(__m128i *a1, __m128i *a2, unsigned int a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  _DWORD *v8; // rdx
  __int8 v10; // al
  __int64 result; // rax
  int v12; // r13d
  __int8 v13; // al
  _QWORD *v14; // rdi
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // r9
  int v22; // eax
  __int64 v23; // rcx
  unsigned int v25; // [rsp+8h] [rbp-28h]

  v7 = a3;
  v8 = a4;
  *a4 = 0;
  *a2 = _mm_loadu_si128(a1);
  a2[1] = _mm_loadu_si128(a1 + 1);
  a2[2] = _mm_loadu_si128(a1 + 2);
  a2[3] = _mm_loadu_si128(a1 + 3);
  a2[4] = _mm_loadu_si128(a1 + 4);
  a2[5] = _mm_loadu_si128(a1 + 5);
  a2[6] = _mm_loadu_si128(a1 + 6);
  a2[7] = _mm_loadu_si128(a1 + 7);
  a2[8] = _mm_loadu_si128(a1 + 8);
  v10 = a1[1].m128i_i8[0];
  if ( v10 == 2 )
  {
    a2[9] = _mm_loadu_si128(a1 + 9);
    a2[10] = _mm_loadu_si128(a1 + 10);
    a2[11] = _mm_loadu_si128(a1 + 11);
    a2[12] = _mm_loadu_si128(a1 + 12);
    a2[13] = _mm_loadu_si128(a1 + 13);
    a2[14] = _mm_loadu_si128(a1 + 14);
    a2[15] = _mm_loadu_si128(a1 + 15);
    a2[16] = _mm_loadu_si128(a1 + 16);
    a2[17] = _mm_loadu_si128(a1 + 17);
    a2[18] = _mm_loadu_si128(a1 + 18);
    a2[19] = _mm_loadu_si128(a1 + 19);
    a2[20] = _mm_loadu_si128(a1 + 20);
    a2[21] = _mm_loadu_si128(a1 + 21);
  }
  else if ( v10 == 5 || v10 == 1 )
  {
    a2[9].m128i_i64[0] = a1[9].m128i_i64[0];
  }
  a2[5].m128i_i64[1] = 0;
  result = a1[1].m128i_u8[0];
  switch ( a1[1].m128i_i8[0] )
  {
    case 0:
    case 3:
    case 4:
    case 6:
      return result;
    case 1:
      v12 = 0;
      v13 = a1[1].m128i_i8[1];
      if ( dword_4F077C4 == 2 && v13 == 2 )
      {
        v25 = a5;
        v22 = sub_8D3A70(a1->m128i_i64[0]);
        v7 = (unsigned int)v7;
        v8 = a4;
        a5 = v25;
        v12 = v22;
        if ( v22 )
        {
          v12 = 1;
          sub_6FF940(a1, (unsigned int)v7, (__int64)a4, v23, v25, a6);
          v13 = a1[1].m128i_i8[1];
          a5 = v25;
          v8 = a4;
          v7 = (unsigned int)v7;
        }
        else
        {
          v13 = a1[1].m128i_i8[1];
        }
      }
      v14 = (_QWORD *)a1[9].m128i_i64[0];
      if ( v13 == 1 )
        result = sub_6E3360(
                   (__int64)v14,
                   v7,
                   (__int64 (__fastcall *)(__int64, _QWORD, _DWORD *, _QWORD))sub_6EC7D0,
                   v8,
                   a5,
                   a6);
      else
        result = (__int64)sub_6EC7D0(v14, v7, v8, a5);
      a2[9].m128i_i64[0] = result;
      if ( v12 )
      {
        v15 = sub_73DCD0(a1[9].m128i_i64[0]);
        v18 = sub_6ED3D0(v15, 0, 0, (__int64)a1[4].m128i_i64 + 4, v16, v17);
        a1[9].m128i_i64[0] = (__int64)v18;
        a1->m128i_i64[0] = *v18;
        v19 = sub_73DCD0(a2[9].m128i_i64[0]);
        result = (__int64)sub_6ED3D0(v19, 0, 0, (__int64)a1[4].m128i_i64 + 4, v20, v21);
        a2[9].m128i_i64[0] = result;
      }
      break;
    case 2:
      a1[18].m128i_i64[0] = 0;
      break;
    default:
      sub_721090(a1);
  }
  return result;
}
