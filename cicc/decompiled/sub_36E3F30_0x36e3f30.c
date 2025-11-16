// Function: sub_36E3F30
// Address: 0x36e3f30
//
__int64 __fastcall sub_36E3F30(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  const __m128i *v5; // rcx
  __int64 v6; // rdx
  _QWORD *v7; // rax
  unsigned int v8; // r13d
  __int64 v10; // r8
  __m128i v11; // xmm1
  _QWORD *v12; // rdi
  int v13; // esi
  __m128i v14; // xmm2
  unsigned __int64 v15; // rcx
  __int64 v16; // r13
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdi
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  int v22; // [rsp+8h] [rbp-58h]
  _OWORD v23[5]; // [rsp+10h] [rbp-50h] BYREF

  v4 = *(_QWORD *)(a2 + 80);
  v21 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v21, v4, 1);
  v5 = *(const __m128i **)(a2 + 40);
  v22 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(v5[7].m128i_i64[1] + 96);
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v8 = 0;
  if ( (unsigned int)v7 <= 3 )
  {
    v10 = *(unsigned int *)(a2 + 68);
    v11 = _mm_loadu_si128(v5 + 10);
    v12 = *(_QWORD **)(a1 + 64);
    v13 = dword_4501120[(unsigned int)v7];
    v23[0] = _mm_loadu_si128(v5 + 5);
    v23[1] = v11;
    v14 = _mm_loadu_si128(v5);
    v15 = *(_QWORD *)(a2 + 48);
    v23[2] = v14;
    v16 = sub_33E66D0(v12, v13, (__int64)&v21, v15, v10, (__int64)&v21, (unsigned __int64 *)v23, 3);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v16, v17, v18, v19);
    v20 = v16;
    v8 = 1;
    sub_3421DB0(v20);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  }
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
  return v8;
}
