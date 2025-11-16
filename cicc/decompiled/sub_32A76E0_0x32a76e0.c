// Function: sub_32A76E0
// Address: 0x32a76e0
//
bool __fastcall sub_32A76E0(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  __int64 *v5; // rdx
  int v6; // eax
  __int64 v7; // rcx
  __int64 v8; // r15
  int v9; // r13d
  int v10; // esi
  int v11; // r14d
  __int64 v12; // rax
  char v13; // al
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rax
  int v20; // edx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // edx
  __int64 v25; // rax
  char v26; // al
  __int64 v27; // rax
  _DWORD *v28; // rax
  __int64 v29; // r14
  __int64 v30; // rcx
  __int64 v31; // rdx
  int v32; // r13d
  __int64 v33; // rax
  int v34; // edx
  __int64 v35; // rax
  __int64 v36; // [rsp-80h] [rbp-80h]
  __m128i v37; // [rsp-78h] [rbp-78h]
  __m128i v38; // [rsp-68h] [rbp-68h]
  __m128i v39; // [rsp-58h] [rbp-58h]
  __m128i v40; // [rsp-48h] [rbp-48h]

  if ( *(_DWORD *)a3 != *(_DWORD *)(a1 + 24) )
    return 0;
  v5 = *(__int64 **)(a1 + 40);
  v6 = *(_DWORD *)(a3 + 8);
  v7 = *v5;
  if ( v6 == *(_DWORD *)(*v5 + 24) )
  {
    v11 = *((_DWORD *)v5 + 2);
    v36 = *v5;
    v12 = *(_QWORD *)(a3 + 16);
    v40 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v7 + 40));
    *(_QWORD *)v12 = v40.m128i_i64[0];
    *(_DWORD *)(v12 + 8) = v40.m128i_i32[2];
    v13 = sub_32657E0(a3 + 24, *(_QWORD *)(*(_QWORD *)(v7 + 40) + 40LL));
    v14 = v36;
    if ( v13
      || (v25 = *(_QWORD *)(a3 + 16),
          v39 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v36 + 40) + 40LL)),
          *(_QWORD *)v25 = v39.m128i_i64[0],
          *(_DWORD *)(v25 + 8) = v39.m128i_i32[2],
          v26 = sub_32657E0(a3 + 24, **(_QWORD **)(v36 + 40)),
          v14 = v36,
          v26) )
    {
      v15 = *(_QWORD *)(a1 + 40);
      v8 = *(_QWORD *)(v15 + 40);
      v9 = *(_DWORD *)(v15 + 48);
      v10 = *(_DWORD *)(v8 + 24);
      if ( *(_BYTE *)(a3 + 44) && *(_DWORD *)(a3 + 40) != (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v14 + 28)) )
        goto LABEL_31;
      v16 = *(_QWORD *)(v14 + 56);
      v17 = 1;
      if ( !v16 )
        goto LABEL_31;
      do
      {
        if ( v11 == *(_DWORD *)(v16 + 8) )
        {
          if ( !v17 )
            goto LABEL_31;
          v16 = *(_QWORD *)(v16 + 32);
          if ( !v16 )
            goto LABEL_32;
          if ( v11 == *(_DWORD *)(v16 + 8) )
            goto LABEL_31;
          v17 = 0;
        }
        v16 = *(_QWORD *)(v16 + 32);
      }
      while ( v16 );
      if ( v17 == 1 )
        goto LABEL_31;
LABEL_32:
      if ( *(_DWORD *)(a3 + 48) != v10
        || (v21 = *(_QWORD *)(a3 + 56), v22 = *(_QWORD *)(v8 + 40), *(_QWORD *)v22 != *(_QWORD *)v21)
        || *(_DWORD *)(v22 + 8) != *(_DWORD *)(v21 + 8) )
      {
LABEL_31:
        v6 = *(_DWORD *)(a3 + 8);
        goto LABEL_5;
      }
      if ( (unsigned __int8)sub_32657E0(a3 + 64, *(_QWORD *)(v22 + 40))
        && (!*(_BYTE *)(a3 + 84) || *(_DWORD *)(a3 + 80) == (*(_DWORD *)(a3 + 80) & *(_DWORD *)(v8 + 28))) )
      {
        v23 = *(_QWORD *)(v8 + 56);
        v24 = 1;
        if ( v23 )
        {
          do
          {
            if ( *(_DWORD *)(v23 + 8) == v9 )
            {
              if ( !v24 )
                goto LABEL_50;
              v23 = *(_QWORD *)(v23 + 32);
              if ( !v23 )
                goto LABEL_47;
              if ( *(_DWORD *)(v23 + 8) == v9 )
                goto LABEL_50;
              v24 = 0;
            }
            v23 = *(_QWORD *)(v23 + 32);
          }
          while ( v23 );
          if ( v24 != 1 )
            goto LABEL_47;
        }
      }
    }
LABEL_50:
    v27 = *(_QWORD *)(a1 + 40);
    v8 = *(_QWORD *)(v27 + 40);
    v9 = *(_DWORD *)(v27 + 48);
    v6 = *(_DWORD *)(a3 + 8);
    v10 = *(_DWORD *)(v8 + 24);
    goto LABEL_5;
  }
  v8 = v5[5];
  v9 = *((_DWORD *)v5 + 12);
  v10 = *(_DWORD *)(v8 + 24);
LABEL_5:
  if ( v6 != v10 )
    return 0;
  v18 = *(_QWORD *)(a3 + 16);
  v38 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(v8 + 40));
  *(_QWORD *)v18 = v38.m128i_i64[0];
  *(_DWORD *)(v18 + 8) = v38.m128i_i32[2];
  if ( !(unsigned __int8)sub_32657E0(a3 + 24, *(_QWORD *)(*(_QWORD *)(v8 + 40) + 40LL)) )
  {
    v35 = *(_QWORD *)(a3 + 16);
    v37 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(v8 + 40) + 40LL));
    *(_QWORD *)v35 = v37.m128i_i64[0];
    *(_DWORD *)(v35 + 8) = v37.m128i_i32[2];
    if ( !(unsigned __int8)sub_32657E0(a3 + 24, **(_QWORD **)(v8 + 40)) )
      return 0;
  }
  if ( *(_BYTE *)(a3 + 44) && *(_DWORD *)(a3 + 40) != (*(_DWORD *)(a3 + 40) & *(_DWORD *)(v8 + 28)) )
    return 0;
  v19 = *(_QWORD *)(v8 + 56);
  if ( !v19 )
    return 0;
  v20 = 1;
  do
  {
    if ( v9 == *(_DWORD *)(v19 + 8) )
    {
      if ( !v20 )
        return 0;
      v19 = *(_QWORD *)(v19 + 32);
      if ( !v19 )
        goto LABEL_52;
      if ( *(_DWORD *)(v19 + 8) == v9 )
        return 0;
      v20 = 0;
    }
    v19 = *(_QWORD *)(v19 + 32);
  }
  while ( v19 );
  if ( v20 == 1 )
    return 0;
LABEL_52:
  v28 = *(_DWORD **)(a1 + 40);
  v29 = *(_QWORD *)v28;
  if ( *(_DWORD *)(a3 + 48) != *(_DWORD *)(*(_QWORD *)v28 + 24LL) )
    return 0;
  v30 = *(_QWORD *)(a3 + 56);
  v31 = *(_QWORD *)(v29 + 40);
  if ( *(_QWORD *)v31 != *(_QWORD *)v30 )
    return 0;
  if ( *(_DWORD *)(v31 + 8) != *(_DWORD *)(v30 + 8) )
    return 0;
  v32 = v28[2];
  if ( !(unsigned __int8)sub_32657E0(a3 + 64, *(_QWORD *)(v31 + 40))
    || *(_BYTE *)(a3 + 84) && *(_DWORD *)(a3 + 80) != (*(_DWORD *)(a3 + 80) & *(_DWORD *)(v29 + 28)) )
  {
    return 0;
  }
  v33 = *(_QWORD *)(v29 + 56);
  if ( !v33 )
    return 0;
  v34 = 1;
  do
  {
    if ( v32 == *(_DWORD *)(v33 + 8) )
    {
      if ( !v34 )
        return 0;
      v33 = *(_QWORD *)(v33 + 32);
      if ( !v33 )
        goto LABEL_47;
      if ( v32 == *(_DWORD *)(v33 + 8) )
        return 0;
      v34 = 0;
    }
    v33 = *(_QWORD *)(v33 + 32);
  }
  while ( v33 );
  if ( v34 == 1 )
    return 0;
LABEL_47:
  result = 1;
  if ( *(_BYTE *)(a3 + 92) )
    return (*(_DWORD *)(a3 + 88) & *(_DWORD *)(a1 + 28)) == *(_DWORD *)(a3 + 88);
  return result;
}
