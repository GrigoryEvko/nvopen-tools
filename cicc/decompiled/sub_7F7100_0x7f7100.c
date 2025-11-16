// Function: sub_7F7100
// Address: 0x7f7100
//
__m128i *__fastcall sub_7F7100(__int64 a1, __int64 a2, int a3)
{
  __m128i *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rsi
  __int64 v11; // r12
  __m128i *result; // rax
  __int64 v13; // rsi
  __int64 i; // r14
  __int64 *v15; // rax
  __int64 *v16; // r12
  __int64 v17; // rdi
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rax

  if ( !a2 )
    return (__m128i *)a1;
  v5 = (__m128i *)sub_7F7100(a1, *(_QWORD *)a2);
  v10 = *(__int64 **)(a2 + 24);
  v11 = (__int64)v5;
  if ( v10 )
    return (__m128i *)sub_7E7E70((__int64)v5, v10);
  v13 = *(_QWORD *)(a2 + 32);
  if ( v13 )
    return sub_7E8750(v5, v13, 0);
  if ( *(_BYTE *)(a2 + 40) )
  {
    for ( i = v5->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v5[1].m128i_i64[0] = (__int64)sub_73A8E0(*(_QWORD *)(a2 + 16), byte_4F06A51[0]);
    result = (__m128i *)sub_73DC30(0x5Du, *(_QWORD *)(i + 160), v11);
    if ( a3 )
      return (__m128i *)sub_7F53E0((__int64)result);
  }
  else
  {
    v15 = (__int64 *)sub_7E2230(v5, 0, v6, v7, v8, v9);
    v16 = v15;
    if ( a3 )
      v16 = (__int64 *)sub_7F53E0((__int64)v15);
    v17 = *(_QWORD *)(a2 + 16);
    if ( v17 )
    {
      v18 = sub_73A8E0(v17, byte_4F06A51[0]);
      v19 = *v16;
      v16[2] = (__int64)v18;
      v20 = sub_8D46C0(v19);
      return (__m128i *)sub_73DC30(0x5Cu, v20, (__int64)v16);
    }
    else
    {
      return (__m128i *)sub_73DCD0(v16);
    }
  }
  return result;
}
