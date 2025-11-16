// Function: sub_26A3510
// Address: 0x26a3510
//
__int64 __fastcall sub_26A3510(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned int *a4)
{
  __int64 v5; // rdi
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v11; // rcx
  _BYTE *v12; // rax
  __m128i v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  int v16; // eax
  int v17; // ecx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // eax
  int v24; // r9d
  int v25; // eax
  int v26; // r9d
  __m128i v27; // [rsp+20h] [rbp-50h] BYREF
  _OWORD v28[4]; // [rsp+30h] [rbp-40h] BYREF

  v5 = *(_QWORD *)(a3 - 32);
  if ( !v5 || *(_BYTE *)v5 || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a3 + 80) )
    goto LABEL_4;
  v8 = (int)*a4;
  v9 = *(_QWORD *)(a2 + 208);
  v11 = v9 + 72 * v8;
  if ( v5 == *(_QWORD *)(v9 + 160LL * *(int *)(v11 + 34644) + 3632) )
  {
    BYTE8(v28[0]) = 0;
    return *(_QWORD *)&v28[0];
  }
  if ( v5 != *(_QWORD *)(v9 + 160LL * *(int *)(v11 + 34640) + 3632) )
  {
    if ( !sub_B2FC80(v5) )
    {
      sub_250D230((unsigned __int64 *)v28, a3, 3, 0);
      v12 = (_BYTE *)sub_26A2E60(a2, *(__int64 *)&v28[0], *((__int64 *)&v28[0] + 1), a1, 0);
      if ( v12[97] )
      {
        v13.m128i_i64[0] = (*(__int64 (__fastcall **)(_BYTE *, _QWORD))(*(_QWORD *)v12 + 120LL))(v12, *a4);
        v27 = v13;
        if ( !v13.m128i_i8[8]
          || v27.m128i_i64[0]
          && (*(_QWORD *)&v28[0] = v27.m128i_i64[0],
              *((_QWORD *)&v28[0] + 1) = a3,
              sub_250C1E0((unsigned __int8 **)v28, v9)) )
        {
          v28[0] = _mm_loadu_si128(&v27);
          return *(_QWORD *)&v28[0];
        }
      }
    }
    goto LABEL_4;
  }
  v14 = 32 * v8;
  v15 = *(_QWORD *)(a1 + v14 + 112);
  v16 = *(_DWORD *)(a1 + v14 + 128);
  if ( !v16 )
  {
LABEL_4:
    *(_QWORD *)&v28[0] = 0;
    BYTE8(v28[0]) = 1;
    return *(_QWORD *)&v28[0];
  }
  v17 = v16 - 1;
  v18 = (v16 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v19 = (__int64 *)(v15 + 16LL * v18);
  v20 = *v19;
  if ( a3 != *v19 )
  {
    v22 = *v19;
    v23 = v18;
    v24 = 1;
    while ( v22 != -4096 )
    {
      v23 = v17 & (v24 + v23);
      v22 = *(_QWORD *)(v15 + 16LL * v23);
      if ( a3 == v22 )
      {
        v25 = 1;
        while ( v20 != -4096 )
        {
          v26 = v25 + 1;
          v18 = v17 & (v25 + v18);
          v19 = (__int64 *)(v15 + 16LL * v18);
          v20 = *v19;
          if ( v22 == *v19 )
            goto LABEL_17;
          v25 = v26;
        }
        v21 = 0;
        goto LABEL_18;
      }
      ++v24;
    }
    goto LABEL_4;
  }
LABEL_17:
  v21 = v19[1];
LABEL_18:
  *(_QWORD *)&v28[0] = v21;
  BYTE8(v28[0]) = 1;
  return *(_QWORD *)&v28[0];
}
