// Function: sub_743530
// Address: 0x743530
//
_QWORD *__fastcall sub_743530(__m128i *a1, __m128i *a2, __int64 a3, int a4, int *a5, __int64 *a6)
{
  __m128i *v9; // rax
  _QWORD *v10; // rcx
  _QWORD *v11; // r12
  unsigned __int64 v12; // rax
  __int64 v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rdi
  const __m128i *v18; // [rsp+10h] [rbp-40h] BYREF
  __int64 v19[7]; // [rsp+18h] [rbp-38h] BYREF

  v9 = (__m128i *)sub_724DC0();
  v10 = (_QWORD *)a1->m128i_i64[0];
  v11 = 0;
  v18 = v9;
  v12 = sub_7410C0(a1, a2, a3, v10, &a1[1].m128i_i32[3], a4, a5, a6, v9, v19);
  if ( !*a5 )
  {
    v11 = (_QWORD *)v12;
    if ( !v12 )
    {
      v14 = v19[0];
      if ( v19[0] )
      {
        v15 = sub_730690(v19[0]);
        v16 = *(_QWORD *)(v14 + 128);
      }
      else
      {
        v15 = sub_73A720(v18, (__int64)a2);
        v16 = v18[8].m128i_i64[0];
      }
      v11 = v15;
      if ( (unsigned int)sub_8D32E0(v16) )
        *((_BYTE *)v11 + 25) |= 1u;
    }
  }
  sub_724E30((__int64)&v18);
  return v11;
}
