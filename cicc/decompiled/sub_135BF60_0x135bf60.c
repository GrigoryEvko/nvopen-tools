// Function: sub_135BF60
// Address: 0x135bf60
//
unsigned __int64 __fastcall sub_135BF60(__int64 a1, __int64 a2, unsigned __int64 a3, const __m128i *a4)
{
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r12
  __int64 v11; // rsi
  __int128 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rbx
  char v16; // si
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r15
  __int64 v28; // rdi
  int v29; // eax
  int v30; // edx
  __int64 v31; // rdi
  int v32; // eax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax

  v8 = sub_135B4D0(a1, a2);
  v9 = *(_QWORD *)(a1 + 64);
  v10 = (__int64)v8;
  if ( v9 )
  {
    if ( !v8[3] )
    {
      sub_13585C0(v9, a1, v8, a3, a4, 0);
      return *(_QWORD *)(a1 + 64);
    }
    if ( a3 > v8[4] )
      v8[4] = a3;
    v11 = v8[5];
    *(_QWORD *)&v12 = v8[6];
    v13 = *(_QWORD *)(v10 + 56);
    if ( v11 == -8 && !(_QWORD)v12 && !v13 )
    {
      *(__m128i *)(v10 + 40) = _mm_loadu_si128(a4);
      *(_QWORD *)(v10 + 56) = a4[1].m128i_i64[0];
      return *(_QWORD *)(a1 + 64);
    }
    *((_QWORD *)&v12 + 1) = a4->m128i_i64[0];
    if ( v11 != a4->m128i_i64[0] )
      *((_QWORD *)&v12 + 1) = 0;
    if ( a4->m128i_i64[1] != (_QWORD)v12 )
      *(_QWORD *)&v12 = 0;
    if ( a4[1].m128i_i64[0] == v13 )
    {
      if ( (unsigned __int64)v12 | v13 | *((_QWORD *)&v12 + 1) )
        goto LABEL_12;
    }
    else
    {
      v13 = 0;
      if ( v12 != 0 )
      {
LABEL_12:
        *(_QWORD *)(v10 + 40) = *((_QWORD *)&v12 + 1);
        *(_QWORD *)(v10 + 48) = v12;
        *(_QWORD *)(v10 + 56) = v13;
        return *(_QWORD *)(a1 + 64);
      }
    }
    *(_QWORD *)(v10 + 40) = -16;
    *(_QWORD *)(v10 + 48) = 0;
    *(_QWORD *)(v10 + 56) = 0;
    return *(_QWORD *)(a1 + 64);
  }
  if ( !v8[3] )
  {
    v34 = sub_1358D80(a1, a2, a3, a4);
    v14 = v34;
    if ( v34 )
    {
      sub_13585C0(v34, a1, (_QWORD *)v10, a3, a4, 0);
    }
    else
    {
      v36 = sub_22077B0(72);
      if ( v36 )
      {
        *(_QWORD *)(v36 + 16) = 0;
        v37 = 0;
        *(_QWORD *)(v36 + 24) = v36 + 16;
        *(_QWORD *)(v36 + 32) = 0;
        *(_QWORD *)(v36 + 40) = 0;
        *(_QWORD *)(v36 + 48) = 0;
        *(_QWORD *)(v36 + 56) = 0;
        *(_QWORD *)(v36 + 64) = 0;
      }
      else
      {
        v37 = MEMORY[0] & 7;
      }
      v38 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(v36 + 8) = a1 + 8;
      v38 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v36 = v38 | v37;
      *(_QWORD *)(v38 + 8) = v36;
      v39 = *(_QWORD *)(a1 + 8) & 7LL | v36;
      *(_QWORD *)(a1 + 8) = v39;
      sub_13585C0(v39 & 0xFFFFFFFFFFFFFFF8LL, a1, (_QWORD *)v10, a3, a4, 0);
      return *(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL;
    }
    return v14;
  }
  v16 = 0;
  if ( a3 > v8[4] )
  {
    v8[4] = a3;
    v16 = 1;
  }
  v17 = v8[5];
  v18 = v8[6];
  v19 = *(_QWORD *)(v10 + 56);
  if ( v17 == -8 && !v18 && !v19 )
  {
    *(__m128i *)(v10 + 40) = _mm_loadu_si128(a4);
    *(_QWORD *)(v10 + 56) = a4[1].m128i_i64[0];
    goto LABEL_23;
  }
  v20 = a4->m128i_i64[0];
  if ( v17 == a4->m128i_i64[0] )
  {
    v35 = a4[1].m128i_i64[0];
    if ( a4->m128i_i64[1] != v18 )
    {
      v18 = 0;
      if ( v35 == v19 )
      {
        v9 = *(_QWORD *)(v10 + 56);
        if ( v20 )
          goto LABEL_52;
        goto LABEL_58;
      }
      if ( !v20 )
        goto LABEL_22;
LABEL_52:
      *(_QWORD *)(v10 + 40) = v20;
      *(_QWORD *)(v10 + 48) = v18;
      *(_QWORD *)(v10 + 56) = v9;
      goto LABEL_23;
    }
    if ( v35 != v19 )
    {
      if ( v20 )
        goto LABEL_52;
      goto LABEL_37;
    }
    v9 = *(_QWORD *)(v10 + 56);
    if ( v20 )
      goto LABEL_52;
LABEL_61:
    v20 = 0;
    v9 = *(_QWORD *)(v10 + 56);
    if ( v18 )
      goto LABEL_52;
    goto LABEL_58;
  }
  v21 = a4[1].m128i_i64[0];
  if ( a4->m128i_i64[1] == v18 )
  {
    if ( v19 != v21 )
    {
LABEL_37:
      v20 = 0;
      if ( !v18 )
        goto LABEL_22;
      goto LABEL_52;
    }
    goto LABEL_61;
  }
  if ( v19 == v21 )
  {
LABEL_58:
    v20 = 0;
    v18 = 0;
    v9 = *(_QWORD *)(v10 + 56);
    if ( !v19 )
      goto LABEL_22;
    goto LABEL_52;
  }
LABEL_22:
  *(_QWORD *)(v10 + 40) = -16;
  *(_QWORD *)(v10 + 48) = 0;
  *(_QWORD *)(v10 + 56) = 0;
LABEL_23:
  if ( v16 )
    sub_1358D80(a1, a2, a3, a4);
  v22 = sub_13582D0(v10, a1);
  v23 = *(_QWORD *)(v22 + 32);
  v14 = v22;
  if ( v23 )
  {
    v24 = *(_QWORD *)(v23 + 32);
    if ( v24 )
    {
      v25 = sub_1357F10(v24, a1);
      v26 = *(_QWORD *)(v23 + 32);
      v27 = v25;
      if ( v25 != v26 )
      {
        *(_DWORD *)(v25 + 64) = (*(_DWORD *)(v25 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v25 + 64) & 0xF8000000;
        v28 = *(_QWORD *)(v23 + 32);
        v29 = *(_DWORD *)(v28 + 64);
        v30 = (v29 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v28 + 64) = v30 | v29 & 0xF8000000;
        if ( !v30 )
          sub_1357730(v28, a1);
        *(_QWORD *)(v23 + 32) = v27;
        v26 = v27;
      }
      if ( *(_QWORD *)(v14 + 32) == v26 )
      {
        return v26;
      }
      else
      {
        *(_DWORD *)(v26 + 64) = (*(_DWORD *)(v26 + 64) + 1) & 0x7FFFFFF | *(_DWORD *)(v26 + 64) & 0xF8000000;
        v31 = *(_QWORD *)(v14 + 32);
        v32 = *(_DWORD *)(v31 + 64);
        v33 = (v32 + 0x7FFFFFF) & 0x7FFFFFF;
        *(_DWORD *)(v31 + 64) = v33 | v32 & 0xF8000000;
        if ( !v33 )
          sub_1357730(v31, a1);
        *(_QWORD *)(v14 + 32) = v26;
        return v26;
      }
    }
    else
    {
      return *(_QWORD *)(v22 + 32);
    }
  }
  return v14;
}
