// Function: sub_1A9B750
// Address: 0x1a9b750
//
__int64 __fastcall sub_1A9B750(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  unsigned int v5; // esi
  __int64 v6; // r8
  int v7; // r10d
  _QWORD *v8; // r13
  unsigned int v9; // ecx
  _QWORD *v10; // rdx
  __int64 v11; // rdi
  int v12; // eax
  int v13; // edx
  __m128i *v14; // rsi
  __int8 *v15; // rsi
  __int64 *v16; // r11
  __int64 v17; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v18; // [rsp+8h] [rbp-58h] BYREF
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  int v20; // [rsp+18h] [rbp-48h]
  __m128i v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 v22; // [rsp+30h] [rbp-30h]

  v17 = sub_1A9B680(a2, *a1);
  result = sub_1A94B30(v17);
  if ( (_BYTE)result )
    return result;
  v4 = a1[1];
  result = v17;
  v21.m128i_i32[2] = 0;
  v22 = 0;
  v5 = *(_DWORD *)(v4 + 24);
  v21.m128i_i64[0] = v17;
  v19 = v17;
  v20 = 0;
  if ( !v5 )
  {
    ++*(_QWORD *)v4;
LABEL_18:
    v5 *= 2;
    goto LABEL_19;
  }
  v6 = *(_QWORD *)(v4 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
  v10 = (_QWORD *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v17 == *v10 )
    return result;
  while ( v11 != -8 )
  {
    if ( v8 || v11 != -16 )
      v10 = v8;
    v9 = (v5 - 1) & (v7 + v9);
    v16 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v16;
    if ( v17 == *v16 )
      return result;
    ++v7;
    v8 = v10;
    v10 = (_QWORD *)(v6 + 16LL * v9);
  }
  v12 = *(_DWORD *)(v4 + 16);
  if ( !v8 )
    v8 = v10;
  ++*(_QWORD *)v4;
  v13 = v12 + 1;
  if ( 4 * (v12 + 1) >= 3 * v5 )
    goto LABEL_18;
  if ( v5 - *(_DWORD *)(v4 + 20) - v13 <= v5 >> 3 )
  {
LABEL_19:
    sub_177C7D0(v4, v5);
    sub_190E590(v4, &v19, &v18);
    v8 = v18;
    v13 = *(_DWORD *)(v4 + 16) + 1;
  }
  *(_DWORD *)(v4 + 16) = v13;
  if ( *v8 != -8 )
    --*(_DWORD *)(v4 + 20);
  *v8 = v19;
  *((_DWORD *)v8 + 2) = v20;
  v14 = *(__m128i **)(v4 + 40);
  if ( v14 == *(__m128i **)(v4 + 48) )
  {
    sub_1A95530((const __m128i **)(v4 + 32), v14, &v21);
    v15 = *(__int8 **)(v4 + 40);
  }
  else
  {
    if ( v14 )
    {
      *v14 = _mm_loadu_si128(&v21);
      v14[1].m128i_i64[0] = v22;
      v14 = *(__m128i **)(v4 + 40);
    }
    v15 = &v14[1].m128i_i8[8];
    *(_QWORD *)(v4 + 40) = v15;
  }
  *((_DWORD *)v8 + 2) = -1431655765 * ((__int64)&v15[-*(_QWORD *)(v4 + 32)] >> 3) - 1;
  return sub_12A9700(a1[2], &v17);
}
