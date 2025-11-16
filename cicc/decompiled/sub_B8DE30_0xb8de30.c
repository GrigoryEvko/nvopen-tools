// Function: sub_B8DE30
// Address: 0xb8de30
//
__int64 __fastcall sub_B8DE30(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  unsigned __int8 v3; // al
  _QWORD *v4; // rdx
  __int64 v5; // r12
  unsigned __int8 v6; // al
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r15
  __m128i v10; // rax
  __int64 result; // rax
  unsigned int v12; // esi
  int v13; // eax
  __m128i *v14; // rdx
  __m128i *v15; // [rsp+0h] [rbp-60h] BYREF
  __m128i *v16; // [rsp+8h] [rbp-58h] BYREF
  __m128i v17; // [rsp+10h] [rbp-50h] BYREF
  __m128i v18[4]; // [rsp+20h] [rbp-40h] BYREF

  v2 = a2 - 16;
  v3 = *(_BYTE *)(a2 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD **)(a2 - 32);
  else
    v4 = (_QWORD *)(v2 - 8LL * ((v3 >> 2) & 0xF));
  v5 = sub_B91420(*v4, a2);
  v6 = *(_BYTE *)(a2 - 16);
  v8 = v7;
  if ( (v6 & 2) != 0 )
    v9 = *(_QWORD *)(a2 - 32);
  else
    v9 = v2 - 8LL * ((v6 >> 2) & 0xF);
  v10.m128i_i64[0] = sub_B91420(*(_QWORD *)(v9 + 8), a2);
  v17.m128i_i64[0] = v5;
  v17.m128i_i64[1] = v8;
  v18[0] = v10;
  result = sub_B8D650(a1, &v17, &v15);
  if ( (_BYTE)result )
    return result;
  v12 = *(_DWORD *)(a1 + 24);
  v13 = *(_DWORD *)(a1 + 16);
  v14 = v15;
  ++*(_QWORD *)a1;
  result = (unsigned int)(v13 + 1);
  v16 = v14;
  if ( 4 * (int)result >= 3 * v12 )
  {
    v12 *= 2;
  }
  else if ( v12 - *(_DWORD *)(a1 + 20) - (unsigned int)result > v12 >> 3 )
  {
    goto LABEL_11;
  }
  sub_B8DC60(a1, v12);
  sub_B8D650(a1, &v17, &v16);
  v14 = v16;
  result = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
LABEL_11:
  *(_DWORD *)(a1 + 16) = result;
  if ( v14->m128i_i64[0] != -1 || v14[1].m128i_i64[0] != -1 )
    --*(_DWORD *)(a1 + 20);
  *v14 = _mm_loadu_si128(&v17);
  v14[1] = _mm_loadu_si128(v18);
  return result;
}
