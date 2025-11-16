// Function: sub_329DEC0
// Address: 0x329dec0
//
bool __fastcall sub_329DEC0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  _QWORD *v6; // rcx
  int v7; // eax
  __int64 v8; // rdx
  __int64 v9; // r13
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r13
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 *v18; // rcx
  __int64 v19; // rdi
  __int64 v20; // r8
  int v21; // r10d
  __int64 v22; // r9
  int v23; // ecx
  __int64 v24; // rcx
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rsi
  __int64 v28; // r8
  __int64 v29; // r9
  int v30; // r11d
  __int64 v31; // r10
  int v32; // esi
  __int64 v33; // rsi
  __int64 v34; // r14
  __int64 v35; // r11
  __m128i v36; // [rsp-68h] [rbp-68h]
  __m128i v37; // [rsp-58h] [rbp-58h]
  __m128i v38; // [rsp-48h] [rbp-48h]
  __m128i v39; // [rsp-38h] [rbp-38h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v6 = *(_QWORD **)(a1 + 40);
  v7 = *(_DWORD *)(a3 + 8);
  v8 = *v6;
  if ( v7 == *(_DWORD *)(*v6 + 24LL) )
  {
    v11 = *(_QWORD *)(a3 + 16);
    v39 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v8 + 40));
    *(_QWORD *)v11 = v39.m128i_i64[0];
    *(_DWORD *)(v11 + 8) = v39.m128i_i32[2];
    v12 = *(_QWORD *)(a3 + 24);
    v38 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v8 + 40) + 40LL));
    *(_QWORD *)v12 = v38.m128i_i64[0];
    *(_DWORD *)(v12 + 8) = v38.m128i_i32[2];
    if ( !*(_BYTE *)(a3 + 36) || *(_DWORD *)(a3 + 32) == (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v8 + 28)) )
    {
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
      v10 = *(_DWORD *)(v9 + 24);
      if ( *(_DWORD *)(a3 + 40) != v10 )
        goto LABEL_28;
      v25 = *(__int64 **)(v9 + 40);
      v26 = *v25;
      if ( *(_DWORD *)(a3 + 48) != *(_DWORD *)(*v25 + 24) )
        goto LABEL_28;
      v27 = *(__int64 **)(v26 + 40);
      v28 = *(_QWORD *)(a3 + 56);
      v29 = *v27;
      v30 = *((_DWORD *)v27 + 2);
      v31 = v27[5];
      v32 = *((_DWORD *)v27 + 12);
      if ( v29 != *(_QWORD *)v28
        || v30 != *(_DWORD *)(v28 + 8)
        || (v34 = *(_QWORD *)(a3 + 64), *(_QWORD *)v34 != v31)
        || v32 != *(_DWORD *)(v34 + 8) )
      {
        if ( *(_QWORD *)v28 != v31 )
          goto LABEL_28;
        if ( v32 != *(_DWORD *)(v28 + 8) )
          goto LABEL_28;
        v33 = *(_QWORD *)(a3 + 64);
        if ( v29 != *(_QWORD *)v33 || v30 != *(_DWORD *)(v33 + 8) )
          goto LABEL_28;
      }
      if ( !*(_BYTE *)(a3 + 76) || *(_DWORD *)(a3 + 72) == (*(_DWORD *)(a3 + 72) & *(_DWORD *)(v26 + 28)) )
      {
        if ( (unsigned __int8)sub_32657E0(a3 + 80, v25[5])
          && (!*(_BYTE *)(a3 + 100) || *(_DWORD *)(a3 + 96) == (*(_DWORD *)(a3 + 96) & *(_DWORD *)(v9 + 28))) )
        {
          goto LABEL_25;
        }
        v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
        v7 = *(_DWORD *)(a3 + 8);
        v10 = *(_DWORD *)(v9 + 24);
      }
      else
      {
LABEL_28:
        v7 = *(_DWORD *)(a3 + 8);
      }
    }
    else
    {
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 40LL);
      v7 = *(_DWORD *)(a3 + 8);
      v10 = *(_DWORD *)(v9 + 24);
    }
  }
  else
  {
    v9 = v6[5];
    v10 = *(_DWORD *)(v9 + 24);
  }
  if ( v7 != v10 )
    return 0;
  v13 = *(_QWORD *)(a3 + 16);
  v37 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v9 + 40));
  *(_QWORD *)v13 = v37.m128i_i64[0];
  *(_DWORD *)(v13 + 8) = v37.m128i_i32[2];
  v14 = *(_QWORD *)(a3 + 24);
  v36 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v9 + 40) + 40LL));
  *(_QWORD *)v14 = v36.m128i_i64[0];
  *(_DWORD *)(v14 + 8) = v36.m128i_i32[2];
  if ( *(_BYTE *)(a3 + 36) )
  {
    if ( *(_DWORD *)(a3 + 32) != (*(_DWORD *)(a3 + 32) & *(_DWORD *)(v9 + 28)) )
      return 0;
  }
  v15 = **(_QWORD **)(a1 + 40);
  if ( *(_DWORD *)(a3 + 40) != *(_DWORD *)(v15 + 24) )
    return 0;
  v16 = *(__int64 **)(v15 + 40);
  v17 = *v16;
  if ( *(_DWORD *)(a3 + 48) != *(_DWORD *)(*v16 + 24) )
    return 0;
  v18 = *(__int64 **)(v17 + 40);
  v19 = *(_QWORD *)(a3 + 56);
  v20 = *v18;
  v21 = *((_DWORD *)v18 + 2);
  v22 = v18[5];
  v23 = *((_DWORD *)v18 + 12);
  if ( v20 != *(_QWORD *)v19
    || v21 != *(_DWORD *)(v19 + 8)
    || (v35 = *(_QWORD *)(a3 + 64), *(_QWORD *)v35 != v22)
    || v23 != *(_DWORD *)(v35 + 8) )
  {
    if ( *(_QWORD *)v19 != v22 )
      return 0;
    if ( v23 != *(_DWORD *)(v19 + 8) )
      return 0;
    v24 = *(_QWORD *)(a3 + 64);
    if ( v20 != *(_QWORD *)v24 || v21 != *(_DWORD *)(v24 + 8) )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 76) && *(_DWORD *)(a3 + 72) != (*(_DWORD *)(a3 + 72) & *(_DWORD *)(v17 + 28))
    || !(unsigned __int8)sub_32657E0(a3 + 80, v16[5])
    || *(_BYTE *)(a3 + 100) && *(_DWORD *)(a3 + 96) != (*(_DWORD *)(a3 + 96) & *(_DWORD *)(v15 + 28)) )
  {
    return 0;
  }
LABEL_25:
  result = 1;
  if ( *(_BYTE *)(a3 + 108) )
    return (*(_DWORD *)(a3 + 104) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 104);
  return result;
}
