// Function: sub_1E1A9C0
// Address: 0x1e1a9c0
//
__int64 __fastcall sub_1E1A9C0(__int64 a1, __int64 a2, const __m128i *a3)
{
  unsigned __int64 v5; // rcx
  unsigned int v6; // ebx
  __int64 v7; // r15
  __int64 v8; // rax
  _QWORD *v9; // r9
  unsigned __int8 v10; // r14
  __int64 v11; // r10
  __int64 v12; // rsi
  int v13; // eax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  char *v16; // rsi
  _QWORD *v17; // rdi
  int v18; // r8d
  __int64 result; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  __m128i *v22; // r14
  bool v23; // zf
  int v24; // esi
  _BYTE *v25; // rax
  _QWORD *v26; // rax
  __m128i v27; // xmm3
  __int64 v28; // rax
  unsigned __int64 v29; // r14
  _QWORD *v30; // rcx
  _QWORD *v31; // rdi
  __int64 v32; // [rsp+0h] [rbp-90h]
  _QWORD *srca; // [rsp+8h] [rbp-88h]
  _QWORD *src; // [rsp+8h] [rbp-88h]
  _QWORD *srcb; // [rsp+8h] [rbp-88h]
  _QWORD *srcc; // [rsp+8h] [rbp-88h]
  __int64 v37; // [rsp+10h] [rbp-80h]
  __int64 v38; // [rsp+10h] [rbp-80h]
  __int64 v39; // [rsp+10h] [rbp-80h]
  __int64 v40; // [rsp+10h] [rbp-80h]
  char v41; // [rsp+1Fh] [rbp-71h]
  __int64 n; // [rsp+20h] [rbp-70h]
  _OWORD v44[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v45; // [rsp+50h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_DWORD *)(a1 + 40);
  if ( v5 <= (unsigned __int64)a3 && (unsigned __int64)a3 < v5 + 40LL * v6 )
  {
    v27 = _mm_loadu_si128(a3 + 1);
    v28 = a3[2].m128i_i64[0];
    v44[0] = _mm_loadu_si128(a3);
    v45 = v28;
    v44[1] = v27;
    return sub_1E1A9C0(a1, a2, v44);
  }
  if ( a3->m128i_i8[0] || (a3->m128i_i8[3] & 0x20) == 0 )
  {
    if ( **(_WORD **)(a1 + 16) == 1 )
    {
LABEL_7:
      v7 = v6;
      v41 = 0;
      n = 5LL * v6;
    }
    else
    {
      while ( v6 )
      {
        v25 = (_BYTE *)(v5 + 40LL * (v6 - 1));
        if ( *v25 || (v25[3] & 0x20) == 0 )
          goto LABEL_7;
        --v6;
      }
      n = 0;
      v7 = 0;
      v41 = 0;
    }
  }
  else
  {
    v7 = v6;
    v41 = 1;
    n = 5LL * v6;
  }
  v8 = sub_1E15BB0(a1);
  v9 = *(_QWORD **)(a1 + 32);
  v10 = *(_BYTE *)(a1 + 44);
  v11 = v8;
  if ( v9 )
  {
    v12 = *(unsigned int *)(a1 + 40);
    v13 = *(_DWORD *)(a1 + 40);
    if ( v12 != 1LL << v10 )
    {
      if ( v6 == (_DWORD)v12 )
      {
        result = v6 + 1;
        v14 = *(_QWORD **)(a1 + 32);
        *(_DWORD *)(a1 + 40) = result;
        goto LABEL_19;
      }
      v14 = *(_QWORD **)(a1 + 32);
      goto LABEL_12;
    }
    *(_BYTE *)(a1 + 44) = v10 + 1;
  }
  else
  {
    *(_BYTE *)(a1 + 44) = 0;
  }
  src = v9;
  v38 = v11;
  v26 = sub_1E1A7D0(a2 + 232, *(_BYTE *)(a1 + 44), (__int64 *)(a2 + 120));
  v11 = v38;
  v9 = src;
  *(_QWORD *)(a1 + 32) = v26;
  v14 = v26;
  if ( v6 )
  {
    if ( v38 )
    {
      sub_1E69AC0(v38, v26, src, v6);
      v13 = *(_DWORD *)(a1 + 40);
      v11 = v38;
      v9 = src;
      if ( v6 != v13 )
      {
        v15 = v13 - v6;
        v16 = (char *)&src[n];
        v17 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * (5 * v7 + 5));
        goto LABEL_13;
      }
    }
    else
    {
      memmove(v26, src, n * 8);
      v13 = *(_DWORD *)(a1 + 40);
      v9 = src;
      v11 = 0;
      if ( v6 != v13 )
      {
        LODWORD(v15) = v13 - v6;
        v16 = (char *)&src[n];
        v17 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * (5 * v7 + 5));
        goto LABEL_43;
      }
    }
    v14 = *(_QWORD **)(a1 + 32);
    goto LABEL_14;
  }
  v13 = *(_DWORD *)(a1 + 40);
  if ( !v13 )
    goto LABEL_14;
LABEL_12:
  v15 = v13 - v6;
  v16 = (char *)&v9[n];
  v17 = &v14[5 * v7 + 5];
  if ( !v11 )
  {
LABEL_43:
    srcb = v9;
    v39 = v11;
    memmove(v17, v16, 40LL * (unsigned int)v15);
    v13 = *(_DWORD *)(a1 + 40);
    v14 = *(_QWORD **)(a1 + 32);
    v9 = srcb;
    v11 = v39;
    goto LABEL_14;
  }
LABEL_13:
  v37 = v11;
  srca = v9;
  sub_1E69AC0(v11, v17, v16, v15);
  v13 = *(_DWORD *)(a1 + 40);
  v11 = v37;
  v14 = *(_QWORD **)(a1 + 32);
  v9 = srca;
LABEL_14:
  result = (unsigned int)(v13 + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( v9 != v14 && v9 )
  {
    result = v10;
    v20 = *(unsigned int *)(a2 + 240);
    if ( v10 >= (unsigned int)v20 )
    {
      v29 = v10 + 1LL;
      if ( result + 1 < v20 )
      {
        *(_DWORD *)(a2 + 240) = v29;
        v21 = *(_QWORD *)(a2 + 232);
        goto LABEL_18;
      }
      if ( result + 1 > v20 )
      {
        if ( v29 > *(unsigned int *)(a2 + 244) )
        {
          v32 = result;
          srcc = v9;
          v40 = v11;
          sub_16CD150(a2 + 232, (const void *)(a2 + 248), v29, 8, v18, (int)v9);
          v9 = srcc;
          v11 = v40;
          v20 = *(unsigned int *)(a2 + 240);
          result = v32;
        }
        v21 = *(_QWORD *)(a2 + 232);
        v30 = (_QWORD *)(v21 + 8 * v20);
        v31 = (_QWORD *)(v21 + 8 * v29);
        if ( v30 != v31 )
        {
          do
          {
            if ( v30 )
              *v30 = 0;
            ++v30;
          }
          while ( v31 != v30 );
          v21 = *(_QWORD *)(a2 + 232);
        }
        *(_DWORD *)(a2 + 240) = v29;
        goto LABEL_18;
      }
    }
    v21 = *(_QWORD *)(a2 + 232);
LABEL_18:
    *v9 = *(_QWORD *)(v21 + 8 * result);
    *(_QWORD *)(*(_QWORD *)(a2 + 232) + 8 * result) = v9;
    v14 = *(_QWORD **)(a1 + 32);
  }
LABEL_19:
  v22 = (__m128i *)&v14[n];
  if ( &v14[n] )
  {
    *v22 = _mm_loadu_si128(a3);
    v22[1] = _mm_loadu_si128(a3 + 1);
    result = a3[2].m128i_i64[0];
    v22[2].m128i_i64[0] = result;
  }
  v23 = v22->m128i_i8[0] == 0;
  v22[1].m128i_i64[0] = a1;
  if ( v23 )
  {
    v22->m128i_i16[1] &= 0xF00Fu;
    v22[1].m128i_i64[1] = 0;
    if ( v11 )
      result = sub_1E699D0(v11, v22);
    if ( !v41 )
    {
      if ( (v22->m128i_i8[3] & 0x10) == 0 )
      {
        result = *(_QWORD *)(a1 + 16);
        if ( v6 >= *(unsigned __int16 *)(result + 2) )
          return result;
        result = *(_QWORD *)(result + 40);
        v24 = *(_DWORD *)(result + 8 * v7 + 4);
        if ( (v24 & 1) == 0 )
          goto LABEL_28;
        sub_1E16A40(a1, BYTE2(v24), v6);
      }
      result = *(_QWORD *)(a1 + 16);
      if ( v6 >= *(unsigned __int16 *)(result + 2) )
        return result;
      result = *(_QWORD *)(result + 40);
      v24 = *(_DWORD *)(result + 8 * v7 + 4);
LABEL_28:
      if ( (v24 & 2) != 0 )
        v22->m128i_i8[4] |= 4u;
    }
  }
  return result;
}
