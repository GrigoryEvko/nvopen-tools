// Function: sub_35095B0
// Address: 0x35095b0
//
unsigned __int64 __fastcall sub_35095B0(_QWORD *a1, _QWORD *a2)
{
  __int64 v3; // rax
  _WORD *v4; // r12
  unsigned __int64 result; // rax
  __m128i *v6; // rsi
  __int64 v7; // r14
  __int64 v8; // r10
  __int16 *v9; // rax
  __int16 *v10; // rdi
  int v11; // eax
  int v12; // r8d
  unsigned __int16 v13; // cx
  unsigned int v14; // esi
  unsigned int v15; // eax
  __int64 v16; // r15
  _WORD *v17; // rdx
  int v18; // eax
  __int64 v20; // [rsp+8h] [rbp-68h]
  _QWORD *v21; // [rsp+10h] [rbp-60h]
  _WORD *i; // [rsp+18h] [rbp-58h]
  __m128i v23; // [rsp+20h] [rbp-50h] BYREF
  __int64 v24; // [rsp+30h] [rbp-40h]

  v21 = *(_QWORD **)(a1[4] + 32LL);
  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v21 + 16LL) + 200LL))(*(_QWORD *)(*v21 + 16LL));
  v4 = (_WORD *)a2[1];
  v20 = v3;
  result = (unsigned __int64)&v4[a2[2]];
  for ( i = (_WORD *)result; i != v4; ++v4 )
  {
    while ( 1 )
    {
      v7 = (unsigned __int16)*v4;
      v8 = v21[48];
      result = *(_QWORD *)(v8 + 8LL * ((unsigned __int16)*v4 >> 6)) & (1LL << *v4);
      if ( !result )
        break;
LABEL_7:
      if ( i == ++v4 )
        return result;
    }
    v9 = (__int16 *)(*(_QWORD *)(v20 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v20 + 8) + 24 * v7 + 8));
    v10 = v9 + 1;
    v11 = *v9;
    v12 = v7 + v11;
    if ( !(_WORD)v11 )
      goto LABEL_3;
    v13 = v7 + v11;
    v14 = a2[2];
    while ( 1 )
    {
      v15 = *(unsigned __int8 *)(a2[6] + v13);
      if ( v15 < v14 )
      {
        v16 = a2[1];
        while ( 1 )
        {
          v17 = (_WORD *)(v16 + 2LL * v15);
          if ( *v17 == v13 )
            break;
          v15 += 256;
          if ( v14 <= v15 )
            goto LABEL_20;
        }
        if ( v17 != (_WORD *)(2LL * a2[2] + v16) )
        {
          result = *(_QWORD *)(v8 + 8 * ((unsigned __int64)v13 >> 6)) & (1LL << v13);
          if ( !result )
            break;
        }
      }
LABEL_20:
      v18 = *v10++;
      if ( !(_WORD)v18 )
        goto LABEL_3;
      v12 += v18;
      v13 = v12;
    }
    if ( !v10 )
    {
LABEL_3:
      v23.m128i_i32[0] = (unsigned __int16)*v4;
      v23.m128i_i64[1] = -1;
      v24 = -1;
      v6 = (__m128i *)a1[24];
      if ( v6 == (__m128i *)a1[25] )
      {
        result = sub_2E341F0(a1 + 23, v6, &v23);
      }
      else
      {
        if ( v6 )
        {
          *v6 = _mm_loadu_si128(&v23);
          v6[1].m128i_i64[0] = v24;
          v6 = (__m128i *)a1[24];
        }
        result = (unsigned __int64)a1;
        a1[24] = (char *)v6 + 24;
      }
      goto LABEL_7;
    }
  }
  return result;
}
