// Function: sub_326FB20
// Address: 0x326fb20
//
__int64 __fastcall sub_326FB20(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rdx
  __int64 v8; // rax
  unsigned int v9; // r12d
  __int64 v10; // rax
  __int64 v11; // r11
  __int64 v12; // r14
  __int64 v13; // r15
  bool v14; // zf
  int v15; // r9d
  __int64 v16; // rdx
  int v17; // eax
  int v18; // ecx
  _DWORD *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // rcx
  int v22; // ebx
  __int64 v23; // rcx
  int v24; // ebx
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // r13
  int v29; // edx
  __int64 v30; // rax
  int v31; // ebx
  __int64 *v32; // r12
  __int64 v33; // r14
  __int128 v34; // rcx
  unsigned int v35; // r13d
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // r12
  unsigned __int16 *v39; // r9
  __int64 v40; // r15
  unsigned __int16 *v41; // r9
  __int128 v42; // [rsp-98h] [rbp-98h]
  __int128 v43; // [rsp-98h] [rbp-98h]
  __int64 v44; // [rsp-88h] [rbp-88h]
  __int64 v45; // [rsp-80h] [rbp-80h]
  __m128i v46; // [rsp-78h] [rbp-78h]
  __m128i v47; // [rsp-68h] [rbp-68h]
  __m128i v48; // [rsp-68h] [rbp-68h]
  __int128 v49; // [rsp-58h] [rbp-58h]
  __int64 v50; // [rsp-40h] [rbp-40h]

  if ( *(_DWORD *)(a4 + 24) != 57 )
    return 0;
  if ( *(_BYTE *)(a1 + 33) )
  {
    v7 = *(_QWORD *)(a1 + 8);
    v8 = 1;
    if ( (_WORD)a2 != 1 )
    {
      if ( !(_WORD)a2 )
        return 0;
      v8 = (unsigned __int16)a2;
      if ( !*(_QWORD *)(v7 + 8LL * (unsigned __int16)a2 + 112) )
        return 0;
    }
    if ( *(_BYTE *)(v7 + 500 * v8 + 6499) )
      return 0;
  }
  v9 = **(unsigned __int16 **)(a4 + 48);
  v10 = *(_QWORD *)(a4 + 40);
  v50 = *(_QWORD *)(*(_QWORD *)(a4 + 48) + 8LL);
  v11 = *(_QWORD *)v10;
  v12 = *(_QWORD *)v10;
  v13 = *(_QWORD *)(v10 + 8);
  v14 = *(_DWORD *)(*(_QWORD *)v10 + 24LL) == 183;
  v15 = *(_DWORD *)(v10 + 8);
  v49 = (__int128)_mm_loadu_si128((const __m128i *)(v10 + 40));
  v16 = *(_QWORD *)(v10 + 40);
  v17 = *(_DWORD *)(v10 + 48);
  if ( v14 )
  {
    v21 = *(_QWORD *)(v11 + 56);
    if ( v21 )
    {
      v22 = 1;
      do
      {
        if ( *(_DWORD *)(v21 + 8) == v15 )
        {
          if ( !v22 )
            goto LABEL_9;
          v21 = *(_QWORD *)(v21 + 32);
          if ( !v21 )
            goto LABEL_31;
          if ( *(_DWORD *)(v21 + 8) == v15 )
            goto LABEL_9;
          v22 = 0;
        }
        v21 = *(_QWORD *)(v21 + 32);
      }
      while ( v21 );
      if ( v22 == 1 )
        goto LABEL_9;
LABEL_31:
      v25 = *(_QWORD *)(v11 + 40);
      v47 = _mm_loadu_si128((const __m128i *)v25);
      v46 = _mm_loadu_si128((const __m128i *)(v25 + 40));
      if ( *(_QWORD *)v25 == v16 && *(_DWORD *)(v25 + 8) == v17 )
        return sub_326F3F0(a2, a3, v9, v50, v46.m128i_i64[0], v46.m128i_i64[1], v49, *(_QWORD *)a1, a5);
      if ( *(_QWORD *)(v25 + 40) == v16 && *(_DWORD *)(v25 + 48) == v17 )
      {
        v45 = a5;
        v26 = v47.m128i_i64[1];
        v44 = *(_QWORD *)a1;
        v27 = v47.m128i_i64[0];
        v42 = v49;
        return sub_326F3F0(a2, a3, v9, v50, v27, v26, v42, v44, v45);
      }
    }
  }
LABEL_9:
  v18 = *(_DWORD *)(v16 + 24);
  if ( v18 == 182 )
  {
    v23 = *(_QWORD *)(v16 + 56);
    if ( v23 )
    {
      v24 = 1;
      do
      {
        if ( *(_DWORD *)(v23 + 8) == v17 )
        {
          if ( !v24 )
            return 0;
          v23 = *(_QWORD *)(v23 + 32);
          if ( !v23 )
            goto LABEL_38;
          if ( *(_DWORD *)(v23 + 8) == v17 )
            return 0;
          v24 = 0;
        }
        v23 = *(_QWORD *)(v23 + 32);
      }
      while ( v23 );
      if ( v24 == 1 )
        return 0;
LABEL_38:
      v28 = *(_QWORD *)(v16 + 40);
      v48 = _mm_loadu_si128((const __m128i *)v28);
      if ( v11 == *(_QWORD *)v28 && *(_DWORD *)(v28 + 8) == v15 )
      {
        v45 = a5;
        v44 = *(_QWORD *)a1;
        v42 = *(_OWORD *)(v28 + 40);
        goto LABEL_42;
      }
      if ( v11 == *(_QWORD *)(v28 + 40) && *(_DWORD *)(v28 + 48) == v15 )
      {
        v45 = a5;
        v44 = *(_QWORD *)a1;
        v42 = (__int128)v48;
LABEL_42:
        v27 = v12;
        v26 = v13;
        return sub_326F3F0(a2, a3, v9, v50, v27, v26, v42, v44, v45);
      }
    }
    return 0;
  }
  if ( v18 != 216 )
    return 0;
  v19 = *(_DWORD **)(v16 + 40);
  v20 = *(_QWORD *)v19;
  if ( *(_DWORD *)(*(_QWORD *)v19 + 24LL) != 182 )
    return 0;
  v29 = v19[2];
  v30 = *(_QWORD *)(v20 + 56);
  if ( !v30 )
    return 0;
  v31 = 1;
  do
  {
    if ( v29 == *(_DWORD *)(v30 + 8) )
    {
      if ( !v31 )
        return 0;
      v30 = *(_QWORD *)(v30 + 32);
      if ( !v30 )
        goto LABEL_52;
      if ( v29 == *(_DWORD *)(v30 + 8) )
        return 0;
      v31 = 0;
    }
    v30 = *(_QWORD *)(v30 + 32);
  }
  while ( v30 );
  if ( v31 == 1 )
    return 0;
LABEL_52:
  v32 = *(__int64 **)(v20 + 40);
  v33 = *v32;
  v34 = *(_OWORD *)v32;
  v35 = *((_DWORD *)v32 + 2);
  v36 = v32[5];
  v37 = v32[6];
  if ( *(_DWORD *)(*v32 + 24) == 214 )
  {
    v40 = *(_QWORD *)(v33 + 40);
    if ( v11 == *(_QWORD *)v40 && *(_DWORD *)(v40 + 8) == v15 )
    {
      v41 = (unsigned __int16 *)(*(_QWORD *)(v33 + 48) + 16LL * v35);
      *((_QWORD *)&v43 + 1) = v32[6];
      *(_QWORD *)&v43 = v36;
      return sub_326F3F0(a2, a3, *v41, *((_QWORD *)v41 + 1), v34, *((__int64 *)&v34 + 1), v43, *(_QWORD *)a1, a5);
    }
  }
  if ( *(_DWORD *)(v36 + 24) != 214 )
    return 0;
  v38 = *(_QWORD *)(v36 + 40);
  if ( v11 != *(_QWORD *)v38 || *(_DWORD *)(v38 + 8) != v15 )
    return 0;
  v39 = (unsigned __int16 *)(*(_QWORD *)(v33 + 48) + 16LL * v35);
  return sub_326F3F0(a2, a3, *v39, *((_QWORD *)v39 + 1), v36, v37, v34, *(_QWORD *)a1, a5);
}
