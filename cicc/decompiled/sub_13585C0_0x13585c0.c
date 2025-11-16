// Function: sub_13585C0
// Address: 0x13585c0
//
__int64 __fastcall sub_13585C0(__int64 a1, __int64 a2, _QWORD *a3, unsigned __int64 a4, const __m128i *a5, char a6)
{
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rcx
  _QWORD *v16; // rax
  _QWORD *v17; // rbx
  int v18; // edx
  __int64 result; // rax
  _QWORD *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int128 v30; // rax
  __int64 v31; // rsi
  _QWORD *v32; // [rsp+8h] [rbp-98h]
  _QWORD v33[6]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v34[2]; // [rsp+40h] [rbp-60h] BYREF
  __m128i v35; // [rsp+50h] [rbp-50h]
  __int64 v36; // [rsp+60h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 67) & 0x40) == 0 && !a6 )
  {
    v20 = *(_QWORD **)(a1 + 16);
    if ( v20 )
    {
      v21 = *a3;
      v34[1] = a4;
      v22 = *(_QWORD *)a2;
      v23 = v20[7];
      v34[0] = v21;
      v24 = a5[1].m128i_i64[0];
      v35 = _mm_loadu_si128(a5);
      v36 = v24;
      v25 = v20[5];
      if ( (v25 == -8 || v25 == -16) && !v20[6] && !v23 )
        v25 = v20[6];
      v26 = *v20;
      v27 = v20[4];
      v33[3] = v20[6];
      v33[4] = v23;
      v32 = v20;
      v33[0] = v26;
      v33[1] = v27;
      v33[2] = v25;
      if ( (unsigned __int8)sub_134CB50(v22, (__int64)v33, (__int64)v34) != 3 )
      {
        *(_BYTE *)(a1 + 67) |= 0x40u;
        *(_DWORD *)(a2 + 56) += *(_DWORD *)(a1 + 68);
        goto LABEL_3;
      }
      if ( a4 > v32[4] )
        v32[4] = a4;
      v29 = v32[5];
      *(_QWORD *)&v30 = v32[6];
      v31 = v32[7];
      if ( v29 == -8 && !(_QWORD)v30 && !v31 )
      {
        *(__m128i *)(v32 + 5) = _mm_loadu_si128(a5);
        v32[7] = a5[1].m128i_i64[0];
        goto LABEL_3;
      }
      *((_QWORD *)&v30 + 1) = a5->m128i_i64[0];
      if ( v29 != a5->m128i_i64[0] )
        *((_QWORD *)&v30 + 1) = 0;
      if ( a5->m128i_i64[1] != (_QWORD)v30 )
        *(_QWORD *)&v30 = 0;
      if ( a5[1].m128i_i64[0] == v31 )
      {
        if ( (unsigned __int64)v30 | v31 | *((_QWORD *)&v30 + 1) )
          goto LABEL_50;
      }
      else
      {
        v31 = 0;
        if ( v30 != 0 )
        {
LABEL_50:
          v32[5] = *((_QWORD *)&v30 + 1);
          v32[6] = v30;
          v32[7] = v31;
          goto LABEL_3;
        }
      }
      v32[5] = -16;
      v32[6] = 0;
      v32[7] = 0;
    }
  }
LABEL_3:
  a3[3] = a1;
  if ( a4 > a3[4] )
    a3[4] = a4;
  v11 = a3[5];
  v12 = a3[6];
  v13 = a3[7];
  if ( v11 == -8 && !v12 && !v13 )
  {
    *(__m128i *)(a3 + 5) = _mm_loadu_si128(a5);
    a3[7] = a5[1].m128i_i64[0];
    goto LABEL_10;
  }
  v14 = a5->m128i_i64[0];
  if ( v11 != a5->m128i_i64[0] )
  {
    v15 = a5[1].m128i_i64[0];
    if ( v12 != a5->m128i_i64[1] )
    {
      if ( v13 != v15 )
      {
LABEL_9:
        a3[5] = -16;
        a3[6] = 0;
        a3[7] = 0;
        goto LABEL_10;
      }
LABEL_39:
      v14 = 0;
      v12 = 0;
      if ( !v13 )
        goto LABEL_9;
      goto LABEL_27;
    }
    if ( v13 != v15 )
      goto LABEL_19;
    goto LABEL_38;
  }
  v28 = a5[1].m128i_i64[0];
  if ( a5->m128i_i64[1] == v12 )
  {
    if ( v13 != v28 )
    {
      v13 = 0;
      if ( v14 )
        goto LABEL_27;
LABEL_19:
      v14 = 0;
      v13 = 0;
      if ( !v12 )
        goto LABEL_9;
      goto LABEL_27;
    }
    if ( v14 )
      goto LABEL_27;
LABEL_38:
    v14 = 0;
    if ( v12 )
      goto LABEL_27;
    goto LABEL_39;
  }
  v12 = 0;
  if ( v13 == v28 )
  {
    if ( v14 )
      goto LABEL_27;
    goto LABEL_39;
  }
  v13 = 0;
  if ( !v14 )
    goto LABEL_9;
LABEL_27:
  a3[5] = v14;
  a3[6] = v12;
  a3[7] = v13;
LABEL_10:
  v16 = *(_QWORD **)(a1 + 24);
  ++*(_DWORD *)(a1 + 68);
  *v16 = a3;
  v17 = a3 + 2;
  *(v17 - 1) = *(_QWORD *)(a1 + 24);
  v18 = *(_DWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 24) = v17;
  result = v18 & 0xF8000000 | (v18 + 1) & 0x7FFFFFF;
  *(_DWORD *)(a1 + 64) = result;
  if ( (v18 & 0x40000000) != 0 )
    ++*(_DWORD *)(a2 + 56);
  return result;
}
