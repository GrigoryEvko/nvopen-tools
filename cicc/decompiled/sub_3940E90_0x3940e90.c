// Function: sub_3940E90
// Address: 0x3940e90
//
__int64 __fastcall sub_3940E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 result; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 *v12; // r14
  __int64 v13; // rcx
  __m128i *v14; // rsi
  unsigned __int64 v15; // rax
  int v16; // r12d
  __m128i **v17; // rbx
  char *v18; // rsi
  __int64 v19; // rdi
  _WORD *v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  _QWORD *v24; // rax
  __m128i *v25; // rdx
  __int64 v26; // rdi
  __m128i si128; // xmm0
  __int64 v28; // rax
  unsigned int v29; // [rsp+8h] [rbp-68h]
  int v30; // [rsp+Ch] [rbp-64h]
  _BYTE *v31; // [rsp+10h] [rbp-60h] BYREF
  __int64 v32; // [rsp+18h] [rbp-58h]
  __m128i v33; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v34[8]; // [rsp+30h] [rbp-40h] BYREF

  v6 = a1;
  result = sub_393F6A0(a1, 2852126720LL, a3, a4, a5, a6);
  v29 = result;
  if ( !(_DWORD)result )
  {
    v10 = *(_QWORD *)(a1 + 72);
    v11 = *(_QWORD *)(a1 + 80);
    v12 = (__int64 *)(a1 + 72);
    v13 = *(_QWORD *)(v10 + 8);
    v14 = (__m128i *)(v11 + 4);
    v15 = *(_QWORD *)(v10 + 16) - v13;
    if ( v15 < v11 + 4 )
    {
      v24 = sub_16E8CB0();
      v25 = (__m128i *)v24[3];
      v26 = (__int64)v24;
      if ( v24[2] - (_QWORD)v25 <= 0x20u )
      {
        v26 = sub_16E7EE0((__int64)v24, "Unexpected end of memory buffer: ", 0x21u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_4530950);
        v25[2].m128i_i8[0] = 32;
        *v25 = si128;
        v25[1] = _mm_load_si128((const __m128i *)&xmmword_3F64F00);
        v24[3] += 33LL;
      }
      v18 = (char *)(*(_QWORD *)(v6 + 80) + 4LL);
      v28 = sub_16E7A90(v26, (__int64)v18);
      v20 = *(_WORD **)(v28 + 24);
      v19 = v28;
      if ( *(_QWORD *)(v28 + 16) - (_QWORD)v20 <= 1u )
      {
        v18 = ".\n";
        sub_16E7EE0(v28, ".\n", 2u);
      }
      else
      {
        *v20 = 2606;
        *(_QWORD *)(v28 + 24) += 2LL;
      }
    }
    else
    {
      *(_QWORD *)(a1 + 80) = v14;
      if ( v15 > v11 )
        v15 = v11;
      v30 = *(_DWORD *)(v13 + v15);
      if ( !v30 )
      {
LABEL_17:
        sub_393D180(a1, (__int64)v14, v11, v13, v8, v9);
        return v29;
      }
      v16 = 0;
      v17 = (__m128i **)(a1 + 88);
      while ( 1 )
      {
        v18 = (char *)&v31;
        v19 = (__int64)v12;
        v31 = 0;
        v32 = 0;
        if ( !(unsigned __int8)sub_393ED40(v12, (__int64 *)&v31) )
          break;
        if ( v31 )
        {
          v33.m128i_i64[0] = (__int64)v34;
          sub_393D750(v33.m128i_i64, v31, (__int64)&v31[v32]);
        }
        else
        {
          v33.m128i_i64[1] = 0;
          v33.m128i_i64[0] = (__int64)v34;
          LOBYTE(v34[0]) = 0;
        }
        v14 = &v33;
        sub_8F9C20(v17, &v33);
        a1 = v33.m128i_i64[0];
        if ( (_QWORD *)v33.m128i_i64[0] != v34 )
        {
          v14 = (__m128i *)(v34[0] + 1LL);
          j_j___libc_free_0(v33.m128i_u64[0]);
        }
        if ( ++v16 == v30 )
          goto LABEL_17;
      }
    }
    sub_393D180(v19, (__int64)v18, (__int64)v20, v21, v22, v23);
    return 4;
  }
  return result;
}
