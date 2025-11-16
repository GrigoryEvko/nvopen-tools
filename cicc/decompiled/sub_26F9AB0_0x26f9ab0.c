// Function: sub_26F9AB0
// Address: 0x26f9ab0
//
__int64 __fastcall sub_26F9AB0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  const __m128i *v8; // r12
  const __m128i *v9; // r13
  __m128i v10; // xmm0
  bool v11; // zf
  _QWORD *v12; // rax
  __int64 result; // rax
  __int64 v14; // r9
  __int64 v15; // r14
  _QWORD *v16; // r14
  _QWORD *v17; // rdi
  __int64 v18; // [rsp+0h] [rbp-80h]
  _BYTE *v19; // [rsp+8h] [rbp-78h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  __int64 v21; // [rsp+20h] [rbp-60h]
  __m128i v23; // [rsp+30h] [rbp-50h] BYREF
  _DWORD *v24; // [rsp+40h] [rbp-40h]

  v6 = a1 + 176;
  v8 = *(const __m128i **)(a2 + 8);
  v9 = *(const __m128i **)a2;
  v18 = a4;
  v19 = a3;
  if ( *(const __m128i **)a2 != v8 )
  {
    while ( 1 )
    {
      v10 = _mm_loadu_si128(v9);
      v11 = *(_BYTE *)(a1 + 204) == 0;
      v23 = v10;
      v24 = (_DWORD *)v9[1].m128i_i64[0];
      if ( v11 )
        goto LABEL_11;
      v12 = *(_QWORD **)(a1 + 184);
      a4 = *(unsigned int *)(a1 + 196);
      a3 = &v12[a4];
      if ( v12 != a3 )
      {
        while ( v10.m128i_i64[1] != *v12 )
        {
          if ( a3 == ++v12 )
            goto LABEL_21;
        }
        goto LABEL_7;
      }
LABEL_21:
      if ( (unsigned int)a4 >= *(_DWORD *)(a1 + 192) )
      {
LABEL_11:
        sub_C8CC70(v6, v10.m128i_i64[1], (__int64)a3, a4, a5, a6);
        if ( (_BYTE)a3 )
          goto LABEL_12;
LABEL_7:
        v9 = (const __m128i *)((char *)v9 + 24);
        if ( v8 == v9 )
          break;
      }
      else
      {
        *(_DWORD *)(a1 + 196) = a4 + 1;
        *a3 = v10.m128i_i64[1];
        ++*(_QWORD *)(a1 + 176);
LABEL_12:
        v15 = sub_ACD640(*(_QWORD *)(v23.m128i_i64[1] + 8), a5, 0);
        if ( *(_BYTE *)(a1 + 104) )
          sub_26F96D0(
            (__int64)&v23,
            "uniform-ret-val",
            15,
            v19,
            v18,
            v14,
            *(__int64 (__fastcall **)(__int64, __int64))(a1 + 112),
            *(_QWORD *)(a1 + 120));
        sub_BD84D0(v23.m128i_i64[1], v15);
        v16 = (_QWORD *)v23.m128i_i64[1];
        if ( *(_BYTE *)v23.m128i_i64[1] == 34 )
        {
          v21 = v23.m128i_i64[1] + 24;
          v20 = *(_QWORD *)(v23.m128i_i64[1] - 96);
          v17 = sub_BD2C40(72, 1u);
          if ( v17 )
            sub_B4C8F0((__int64)v17, v20, 1u, v21, 0);
          sub_AA5980(*(v16 - 8), v16[5], 0);
          v16 = (_QWORD *)v23.m128i_i64[1];
        }
        sub_B43D60(v16);
        if ( !v24 )
          goto LABEL_7;
        v9 = (const __m128i *)((char *)v9 + 24);
        --*v24;
        if ( v8 == v9 )
          break;
      }
    }
  }
  *(_BYTE *)(a2 + 24) = 1;
  result = *(_QWORD *)(a2 + 32);
  if ( result != *(_QWORD *)(a2 + 40) )
    *(_QWORD *)(a2 + 40) = result;
  return result;
}
