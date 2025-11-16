// Function: sub_2E8EAD0
// Address: 0x2e8ead0
//
__int64 __fastcall sub_2E8EAD0(__int64 a1, __int64 a2, const __m128i *a3)
{
  unsigned __int64 v5; // rcx
  __int64 v6; // r14
  unsigned int v7; // ebx
  size_t v8; // r15
  __int64 v9; // rax
  _QWORD *v10; // r9
  unsigned int v11; // r11d
  __int64 v12; // r8
  __int64 v13; // r10
  __int64 result; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rax
  _QWORD *v17; // rsi
  _QWORD *v18; // rdi
  unsigned int v19; // ecx
  unsigned int v20; // ecx
  unsigned __int64 v21; // rcx
  __int64 v22; // rsi
  __m128i *v23; // r15
  bool v24; // zf
  unsigned __int16 *v25; // rdx
  unsigned int v26; // ecx
  _BYTE *v27; // rax
  char v28; // cl
  unsigned int v29; // eax
  __int64 v30; // rdx
  _QWORD *v31; // rax
  __m128i v32; // xmm3
  __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rcx
  unsigned __int16 v36; // si
  __int64 v37; // rax
  char *v38; // rdx
  unsigned __int64 v39; // rdx
  _QWORD *v40; // rcx
  _QWORD *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // [rsp+0h] [rbp-90h]
  __int64 v44; // [rsp+8h] [rbp-88h]
  __int64 v45; // [rsp+8h] [rbp-88h]
  __int64 v46; // [rsp+8h] [rbp-88h]
  _QWORD *v47; // [rsp+8h] [rbp-88h]
  unsigned __int8 v48; // [rsp+8h] [rbp-88h]
  unsigned __int8 v49; // [rsp+10h] [rbp-80h]
  unsigned __int8 v50; // [rsp+10h] [rbp-80h]
  unsigned __int8 v51; // [rsp+10h] [rbp-80h]
  __int64 v52; // [rsp+10h] [rbp-80h]
  _QWORD *v53; // [rsp+10h] [rbp-80h]
  _QWORD *v54; // [rsp+18h] [rbp-78h]
  _QWORD *v55; // [rsp+18h] [rbp-78h]
  _QWORD *v56; // [rsp+18h] [rbp-78h]
  __int64 v57; // [rsp+18h] [rbp-78h]
  __int64 v58; // [rsp+18h] [rbp-78h]
  char v59; // [rsp+27h] [rbp-69h]
  __int64 v60; // [rsp+28h] [rbp-68h]
  _QWORD *v61; // [rsp+28h] [rbp-68h]
  __int64 v62; // [rsp+28h] [rbp-68h]
  __int64 v63; // [rsp+28h] [rbp-68h]
  __int64 v64; // [rsp+28h] [rbp-68h]
  _OWORD v65[2]; // [rsp+30h] [rbp-60h] BYREF
  __int64 v66; // [rsp+50h] [rbp-40h]

  v5 = *(_QWORD *)(a1 + 32);
  v6 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v5 <= (unsigned __int64)a3 && (unsigned __int64)a3 < v5 + 40LL * (unsigned int)v6 )
  {
    v32 = _mm_loadu_si128(a3 + 1);
    v33 = a3[2].m128i_i64[0];
    v65[0] = _mm_loadu_si128(a3);
    v66 = v33;
    v65[1] = v32;
    return sub_2E8EAD0(a1, a2, v65);
  }
  v7 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( a3->m128i_i8[0] || (a3->m128i_i8[3] & 0x20) == 0 )
  {
    if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 > 1 )
    {
      while ( 1 )
      {
        if ( !v7 )
        {
          v59 = 0;
          v8 = 0;
          v6 = 0;
          goto LABEL_8;
        }
        v27 = (_BYTE *)(v5 + 40LL * (v7 - 1));
        if ( *v27 || (v27[3] & 0x20) == 0 )
          break;
        --v7;
      }
      v6 = v7;
    }
    v59 = 0;
    v8 = 5 * v6;
  }
  else
  {
    v6 = (unsigned int)v6;
    v59 = 1;
    v8 = 5LL * (unsigned int)v6;
  }
LABEL_8:
  v9 = sub_2E866D0(a1);
  v10 = *(_QWORD **)(a1 + 32);
  v11 = *(unsigned __int8 *)(a1 + 43);
  v12 = a2;
  v13 = v9;
  if ( v10 )
  {
    LODWORD(result) = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
    if ( (unsigned int)result != 1LL << v11 )
    {
      if ( v7 == (_DWORD)result )
      {
        v15 = *(_QWORD **)(a1 + 32);
        *(_WORD *)(a1 + 40) = v7 + 1;
        result = (unsigned __int64)(v7 + 1) >> 16;
        *(_BYTE *)(a1 + 42) = (v7 + 1) >> 16;
        goto LABEL_19;
      }
      v15 = *(_QWORD **)(a1 + 32);
      goto LABEL_12;
    }
    v28 = v11 + 1;
    *(_BYTE *)(a1 + 43) = v11 + 1;
    v29 = (unsigned __int8)(v11 + 1);
    v30 = (unsigned __int8)(v11 + 1);
  }
  else
  {
    *(_BYTE *)(a1 + 43) = 0;
    v28 = 0;
    v30 = 0;
    v29 = 0;
  }
  if ( *(_DWORD *)(a2 + 240) > v29 && (v31 = (_QWORD *)(*(_QWORD *)(a2 + 232) + 8 * v30), (v15 = (_QWORD *)*v31) != 0) )
  {
    *v31 = *v15;
  }
  else
  {
    v37 = *(_QWORD *)(a2 + 128);
    *(_QWORD *)(a2 + 208) += 40LL << v28;
    v15 = (_QWORD *)((v37 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    v38 = (char *)v15 + (40LL << v28);
    if ( *(_QWORD *)(a2 + 136) >= (unsigned __int64)v38 && v37 )
    {
      *(_QWORD *)(a2 + 128) = v38;
    }
    else
    {
      v48 = v11;
      v53 = v10;
      v58 = v13;
      v42 = sub_9D1E70(a2 + 128, 40LL << v28, 40LL << v28, 3);
      v10 = v53;
      v13 = v58;
      v11 = v48;
      v12 = a2;
      v15 = (_QWORD *)v42;
    }
  }
  *(_QWORD *)(a1 + 32) = v15;
  if ( v7 )
  {
    v46 = v12;
    v51 = v11;
    if ( v13 )
    {
      v56 = v10;
      v63 = v13;
      result = sub_2EBEBD0(v13, v15, v10, v7);
      v13 = v63;
      v10 = v56;
      v11 = v51;
      v19 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      v12 = v46;
      if ( v7 != v19 )
      {
        v17 = &v56[v8];
        LODWORD(v16) = v19 - v7;
        v18 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL * (v7 + 1));
        goto LABEL_13;
      }
    }
    else
    {
      v61 = v10;
      result = (__int64)memmove(v15, v10, v8 * 8);
      v10 = v61;
      v13 = 0;
      v11 = v51;
      v19 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
      v12 = v46;
      if ( v7 != v19 )
      {
        v17 = &v61[v8];
        v16 = v19 - v7;
        v18 = (_QWORD *)(*(_QWORD *)(a1 + 32) + 40LL * (v7 + 1));
        goto LABEL_48;
      }
    }
    v15 = *(_QWORD **)(a1 + 32);
    goto LABEL_14;
  }
  v19 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  result = v19;
  if ( !v19 )
    goto LABEL_14;
LABEL_12:
  v16 = (unsigned int)result - v7;
  v17 = &v10[v8];
  v18 = &v15[5 * v7 + 5];
  if ( !v13 )
  {
LABEL_48:
    v45 = v12;
    v50 = v11;
    v55 = v10;
    v62 = v13;
    result = (__int64)memmove(v18, v17, 40 * v16);
    v15 = *(_QWORD **)(a1 + 32);
    v12 = v45;
    v11 = v50;
    v10 = v55;
    v13 = v62;
    v19 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
    goto LABEL_14;
  }
LABEL_13:
  v44 = v12;
  v49 = v11;
  v54 = v10;
  v60 = v13;
  result = sub_2EBEBD0(v13, v18, v17, (unsigned int)v16);
  v15 = *(_QWORD **)(a1 + 32);
  v13 = v60;
  v10 = v54;
  v11 = v49;
  v12 = v44;
  v19 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
LABEL_14:
  v20 = v19 + 1;
  *(_WORD *)(a1 + 40) = v20;
  *(_BYTE *)(a1 + 42) = BYTE2(v20);
  if ( v10 && v10 != v15 )
  {
    v21 = *(unsigned int *)(v12 + 240);
    result = (unsigned __int8)v11;
    if ( v11 >= (unsigned int)v21 )
    {
      v39 = (unsigned __int8)v11 + 1LL;
      if ( v39 != v21 )
      {
        if ( (unsigned __int64)(unsigned __int8)v11 + 1 >= v21 )
        {
          if ( v39 > *(unsigned int *)(v12 + 244) )
          {
            v43 = (unsigned __int8)v11;
            v47 = v10;
            v52 = v13;
            v57 = v12;
            v64 = (unsigned __int8)v11 + 1LL;
            sub_C8D5F0(v12 + 232, (const void *)(v12 + 248), v39, 8u, v12, (__int64)v10);
            v12 = v57;
            result = v43;
            v10 = v47;
            v13 = v52;
            v21 = *(unsigned int *)(v57 + 240);
            v39 = v64;
          }
          v22 = *(_QWORD *)(v12 + 232);
          v40 = (_QWORD *)(v22 + 8 * v21);
          v41 = (_QWORD *)(v22 + 8 * v39);
          if ( v40 != v41 )
          {
            do
            {
              if ( v40 )
                *v40 = 0;
              ++v40;
            }
            while ( v41 != v40 );
            v22 = *(_QWORD *)(v12 + 232);
          }
          *(_DWORD *)(v12 + 240) = v39;
          goto LABEL_18;
        }
        *(_DWORD *)(v12 + 240) = v39;
      }
    }
    v22 = *(_QWORD *)(v12 + 232);
LABEL_18:
    *v10 = *(_QWORD *)(v22 + 8 * result);
    *(_QWORD *)(*(_QWORD *)(v12 + 232) + 8 * result) = v10;
    v15 = *(_QWORD **)(a1 + 32);
  }
LABEL_19:
  v23 = (__m128i *)&v15[v8];
  if ( v23 )
  {
    *v23 = _mm_loadu_si128(a3);
    v23[1] = _mm_loadu_si128(a3 + 1);
    result = a3[2].m128i_i64[0];
    v23[2].m128i_i64[0] = result;
  }
  v24 = v23->m128i_i8[0] == 0;
  v23[1].m128i_i64[0] = a1;
  if ( v24 )
  {
    v23->m128i_i16[1] &= 0xF00Fu;
    v23[1].m128i_i64[1] = 0;
    if ( v13 )
      sub_2EBEAE0(v13, v23);
    result = v23->m128i_i8[3] & 0x10;
    if ( v59 )
      goto LABEL_29;
    v25 = *(unsigned __int16 **)(a1 + 16);
    v26 = v25[1];
    if ( (_BYTE)result )
    {
      if ( v7 >= v26 )
        return result;
LABEL_27:
      if ( (v25[20 * *v25 + 22 + 3 * v25[8] + 3 * v6] & 2) == 0 )
        goto LABEL_29;
      goto LABEL_28;
    }
    if ( v7 < v26 )
    {
      v34 = v25[8];
      v35 = 4 * (5LL * *v25 + 5);
      v36 = v25[3 * v6 + 2 + 3 * v34 + v35];
      if ( (v36 & 1) != 0 )
      {
        sub_2E89ED0(a1, (unsigned __int8)(v36 >> 4), v7);
        v25 = *(unsigned __int16 **)(a1 + 16);
        result = v23->m128i_i8[3] & 0x10;
        if ( v7 < v25[1] )
          goto LABEL_27;
LABEL_29:
        if ( (_BYTE)result )
          return result;
        goto LABEL_30;
      }
      if ( (v25[3 * v6 + 2 + 3 * v34 + v35] & 2) != 0 )
      {
LABEL_28:
        v23->m128i_i8[4] |= 4u;
        goto LABEL_29;
      }
    }
LABEL_30:
    result = (unsigned int)*(unsigned __int16 *)(a1 + 68) - 14;
    if ( (unsigned __int16)(*(_WORD *)(a1 + 68) - 14) <= 4u )
      v23->m128i_i8[4] |= 8u;
  }
  return result;
}
