// Function: sub_FF0A60
// Address: 0xff0a60
//
__int64 __fastcall sub_FF0A60(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v5; // rax
  __int64 v6; // rbx
  __int64 result; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // r13
  unsigned int i; // ebx
  __int64 v11; // rsi
  __int64 v12; // rax
  _WORD *v13; // rdx
  unsigned __int8 *v14; // rcx
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-50h]
  __int64 v17; // [rsp+8h] [rbp-48h]
  unsigned __int8 *v18; // [rsp+10h] [rbp-40h]
  int v19; // [rsp+18h] [rbp-38h]

  v3 = *(__m128i **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x1Eu )
  {
    sub_CB6200(a2, "---- Branch Probabilities ----\n", 0x1Fu);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CFB0);
    qmemcpy(&v3[1], "abilities ----\n", 15);
    *v3 = si128;
    *(_QWORD *)(a2 + 32) += 31LL;
  }
  v5 = *(_QWORD *)(a1 + 64);
  v6 = *(_QWORD *)(v5 + 80);
  result = v5 + 72;
  v16 = result;
  v17 = v6;
  if ( v6 != result )
  {
    do
    {
      if ( !v17 )
        BUG();
      v8 = *(_QWORD *)(v17 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v8 != v17 + 24 )
      {
        if ( !v8 )
          BUG();
        v9 = v8 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 <= 0xA )
        {
          v19 = sub_B46E30(v9);
          if ( v19 )
          {
            for ( i = 0; i != v19; ++i )
            {
              v12 = sub_B46EC0(v9, i);
              v13 = *(_WORD **)(a2 + 32);
              v14 = (unsigned __int8 *)v12;
              if ( *(_QWORD *)(a2 + 24) - (_QWORD)v13 > 1u )
              {
                v11 = a2;
                *v13 = 8224;
                *(_QWORD *)(a2 + 32) += 2LL;
              }
              else
              {
                v18 = (unsigned __int8 *)v12;
                v15 = sub_CB6200(a2, (unsigned __int8 *)"  ", 2u);
                v14 = v18;
                v11 = v15;
              }
              sub_FF0830(a1, v11, (unsigned __int8 *)(v17 - 24), v14);
            }
          }
        }
      }
      result = *(_QWORD *)(v17 + 8);
      v17 = result;
    }
    while ( v16 != result );
  }
  return result;
}
