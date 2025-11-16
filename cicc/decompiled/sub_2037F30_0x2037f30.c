// Function: sub_2037F30
// Address: 0x2037f30
//
__int64 *__fastcall sub_2037F30(__int64 **a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  char v12; // al
  __int64 *v13; // r13
  unsigned int v14; // edx
  _BYTE v15[8]; // [rsp+0h] [rbp-40h] BYREF
  __int64 v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h]

  v5 = *(_QWORD *)(a2 + 32);
  v6 = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 40LL) + 16LL * *(unsigned int *)(v5 + 48);
  v7 = *(_QWORD *)(*(_QWORD *)v5 + 40LL) + 16LL * *(unsigned int *)(v5 + 8);
  if ( *(_BYTE *)v7 == *(_BYTE *)v6 && (*(_QWORD *)(v7 + 8) == *(_QWORD *)(v6 + 8) || *(_BYTE *)v7) )
    return sub_2036AE0(a1, a2, *(double *)a3.m128i_i64, a4, a5);
  sub_1F40D10(
    (__int64)v15,
    (__int64)*a1,
    a1[1][6],
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v12 = v16;
  v13 = a1[1];
  v15[0] = v16;
  v16 = v17;
  if ( v15[0] )
    v14 = word_4305480[(unsigned __int8)(v12 - 14)];
  else
    v14 = sub_1F58D30((__int64)v15);
  return sub_1D40890(v13, a2, v14, v9, v10, v11, a3, a4, a5);
}
