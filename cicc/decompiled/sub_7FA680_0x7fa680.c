// Function: sub_7FA680
// Address: 0x7fa680
//
__int64 __fastcall sub_7FA680(const __m128i *a1, __int64 m128i_i64, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  int v7; // eax
  int v8; // ebx
  __int64 result; // rax
  const __m128i *v10; // r14
  _QWORD *v11; // r15
  __int8 v12; // cl
  __int64 v13; // r12
  void *v14; // rax
  const __m128i *v15; // rax
  __int64 *v16; // rax
  const __m128i *v17; // rax
  __m128i *v18; // r15
  _BYTE *v19; // rax
  const __m128i *v20; // r12
  __int64 v21; // rsi
  _BYTE *v22; // rax
  const __m128i *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // [rsp+8h] [rbp-48h]
  __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1->m128i_i64[0];
  v29[0] = 0;
  v7 = *(unsigned __int8 *)(v6 + 140);
  for ( LOBYTE(v8) = a1[1].m128i_i8[9] & 1; (_BYTE)v7 == 12; v7 = *(unsigned __int8 *)(v6 + 140) )
    v6 = *(_QWORD *)(v6 + 160);
  result = (unsigned int)(v7 - 9);
  if ( (unsigned __int8)result <= 2u )
  {
    v10 = (const __m128i *)a1[4].m128i_i64[1];
    v11 = (_QWORD *)v10[1].m128i_i64[0];
    if ( (a1[3].m128i_i8[12] & 2) != 0
      && (sub_7E6EE0(a1[4].m128i_i64[1], v10[1].m128i_i64[0])
       || (m128i_i64 = (__int64)v10, sub_7E6EE0((__int64)v11, (__int64)v10))) )
    {
      m128i_i64 = (__int64)v10[1].m128i_i64;
      v8 = (unsigned __int8)v8;
      v11 = sub_7FA3E0((_QWORD **)v11, (const __m128i *)v10[1].m128i_i64, v29);
      if ( (*(_BYTE *)(v6 + 179) & 1) != 0 )
        goto LABEL_6;
    }
    else
    {
      v8 = (unsigned __int8)v8;
      if ( (*(_BYTE *)(v6 + 179) & 1) != 0 )
      {
LABEL_6:
        if ( !v8 )
          v10 = (const __m128i *)sub_731370((__int64)v10, m128i_i64, a3, a4, a5, a6);
        if ( (unsigned int)sub_731770((__int64)v11, 0, a3, a4, a5, a6) )
        {
          v11[2] = v10;
          v12 = v10[1].m128i_i8[9];
          v10[1].m128i_i64[0] = 0;
          result = sub_73D8E0((__int64)a1, 0x5Bu, a1->m128i_i64[0], v12 & 1, (__int64)v11);
        }
        else
        {
          result = sub_730620((__int64)a1, v10);
        }
        if ( (a1[1].m128i_i8[9] & 4) != 0 )
        {
          v13 = sub_72CBE0();
          v14 = sub_730FF0(a1);
          v15 = (const __m128i *)sub_73E110((__int64)v14, v13);
          result = sub_730620((__int64)a1, v15);
        }
LABEL_12:
        if ( v29[0] )
        {
          v16 = (__int64 *)sub_730FF0(a1);
          v17 = (const __m128i *)sub_73DF90(v29[0], v16);
          return sub_730620((__int64)a1, v17);
        }
        return result;
      }
    }
    result = *(_QWORD *)(v6 + 168);
    v28 = *(_QWORD *)(result + 32);
    if ( *(_QWORD *)(v6 + 128) != v28 )
    {
      v10[1].m128i_i64[0] = 0;
      v18 = (__m128i *)sub_7EC130((const __m128i *)v11);
      v19 = sub_73E1B0((__int64)v10, m128i_i64);
      v20 = (const __m128i *)sub_7FA4E0(v19, v18, v28, v6);
      if ( (a1[1].m128i_i8[9] & 4) == 0 )
      {
        v21 = sub_72D2E0(a1->m128i_i64[0]);
        v22 = sub_73E110((__int64)v20, v21);
        v23 = (const __m128i *)sub_73DCD0(v22);
        v20 = v23;
        if ( !v8 )
          v20 = (const __m128i *)sub_731370((__int64)v23, v21, v24, v25, v26, v27);
      }
      result = sub_730620((__int64)a1, v20);
    }
    goto LABEL_12;
  }
  return result;
}
