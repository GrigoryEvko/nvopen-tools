// Function: sub_13779F0
// Address: 0x13779f0
//
__int64 __fastcall sub_13779F0(__int64 a1, __int64 a2)
{
  __m128i *v4; // rdx
  __m128i si128; // xmm0
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // r13
  unsigned int i; // ebx
  __int64 v13; // rsi
  __int64 v14; // rax
  _WORD *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]
  int v21; // [rsp+1Ch] [rbp-34h]

  v4 = *(__m128i **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v4 <= 0x1Eu )
  {
    sub_16E7EE0(a2, "---- Branch Probabilities ----\n", 31);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CFB0);
    qmemcpy(&v4[1], "abilities ----\n", 15);
    *v4 = si128;
    *(_QWORD *)(a2 + 24) += 31LL;
  }
  v6 = *(_QWORD *)(a1 + 64);
  v7 = *(_QWORD *)(v6 + 80);
  result = v6 + 72;
  v18 = result;
  v19 = v7;
  if ( v7 != result )
  {
    do
    {
      v9 = v19 - 24;
      if ( !v19 )
        v9 = 0;
      v10 = sub_157EBA0(v9);
      v11 = v10;
      if ( v10 )
      {
        v21 = sub_15F4D60(v10);
        if ( v21 )
        {
          for ( i = 0; i != v21; ++i )
          {
            v14 = sub_15F4DF0(v11, i);
            v15 = *(_WORD **)(a2 + 24);
            v16 = v14;
            if ( *(_QWORD *)(a2 + 16) - (_QWORD)v15 > 1u )
            {
              v13 = a2;
              *v15 = 8224;
              *(_QWORD *)(a2 + 24) += 2LL;
            }
            else
            {
              v20 = v14;
              v17 = sub_16E7EE0(a2, "  ", 2);
              v16 = v20;
              v13 = v17;
            }
            sub_1377720(a1, v13, v9, v16);
          }
        }
      }
      result = *(_QWORD *)(v19 + 8);
      v19 = result;
    }
    while ( v18 != result );
  }
  return result;
}
