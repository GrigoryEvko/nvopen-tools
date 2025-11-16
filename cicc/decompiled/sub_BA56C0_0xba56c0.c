// Function: sub_BA56C0
// Address: 0xba56c0
//
void __fastcall sub_BA56C0(__m128i *a1, __int64 a2, unsigned __int8 *a3)
{
  __m128i *v3; // r14
  unsigned __int8 v6; // al
  __int64 v7; // rbx
  unsigned __int8 v8; // al
  __int64 v9; // r14
  __int64 v10; // rsi
  unsigned __int8 *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  const __m128i *v18; // r14
  int v19; // r13d
  unsigned int i; // ebx
  __int64 v21; // rdi

  v3 = a1 - 1;
  v6 = a1[-1].m128i_u8[0];
  if ( (v6 & 2) == 0 )
  {
    v7 = (a2 - (__int64)&v3->m128i_i64[-((v6 >> 2) & 0xF)]) >> 3;
    if ( (a1->m128i_i8[1] & 0x7F) == 0 )
      goto LABEL_3;
LABEL_13:
    sub_B97110((__int64)a1, v7, (__int64)a3);
    return;
  }
  v7 = (a2 - a1[-2].m128i_i64[0]) >> 3;
  if ( (a1->m128i_i8[1] & 0x7F) != 0 )
    goto LABEL_13;
LABEL_3:
  sub_B93910((__int64)a1);
  v8 = a1[-1].m128i_u8[0];
  if ( (v8 & 2) != 0 )
    v9 = a1[-2].m128i_i64[0];
  else
    v9 = (__int64)&v3->m128i_i64[-((v8 >> 2) & 0xF)];
  v10 = (unsigned int)v7;
  v11 = *(unsigned __int8 **)(v9 + 8LL * (unsigned int)v7);
  sub_B97110((__int64)a1, v7, (__int64)a3);
  if ( a1 == (__m128i *)a3 || v11 && !a3 && *v11 == 1 )
  {
    if ( (a1->m128i_i8[1] & 0x7F) == 2 || (v13 = a1[-1].m128i_u32[2], (_DWORD)v13) )
      sub_B93190((__int64)a1, (unsigned int)v7, v12, v13, v14);
    goto LABEL_11;
  }
  v18 = sub_BA3380(a1);
  if ( a1 == v18 )
  {
    if ( (a1->m128i_i8[1] & 0x7F) == 2 || a1[-1].m128i_i32[2] )
      sub_B93270((__int64)v18, v11, a3, v16, v17);
  }
  else
  {
    if ( (a1->m128i_i8[1] & 0x7F) != 2 && !a1[-1].m128i_i32[2] )
    {
LABEL_11:
      sub_B95A20((unsigned __int8 *)a1);
      return;
    }
    if ( (a1[-1].m128i_i8[0] & 2) != 0 )
      v19 = a1[-2].m128i_i32[2];
    else
      v19 = ((unsigned __int16)a1[-1].m128i_i16[0] >> 6) & 0xF;
    if ( v19 )
    {
      for ( i = 0; i != v19; ++i )
      {
        v10 = i;
        sub_B97110((__int64)a1, v10, 0);
      }
    }
    v21 = a1->m128i_i64[1];
    if ( (v21 & 4) != 0 )
    {
      v10 = (__int64)v18;
      sub_BA6110(v21 & 0xFFFFFFFFFFFFFFF8LL, v18);
    }
    sub_B97380((__int64)a1, v10, v15, v16, v17);
  }
}
