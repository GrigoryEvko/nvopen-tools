// Function: sub_CF9460
// Address: 0xcf9460
//
__int64 __fastcall sub_CF9460(
        __int64 a1,
        __int64 a2,
        _DWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 (__fastcall *a7)(__int64, __int64, unsigned __int64, __int64, __int64, __int64, __int64, __int64, __int64),
        __int64 a8)
{
  __int64 v8; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 v14; // rsi
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // r15
  _DWORD *v28; // r13
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // r14
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __m128i v34; // xmm0
  __int64 v35; // rsi
  int v37; // esi
  int v38; // r10d
  _DWORD *v39; // [rsp+8h] [rbp-98h]
  __m128i v43; // [rsp+30h] [rbp-70h] BYREF
  __int64 v44; // [rsp+40h] [rbp-60h]
  __m128i v45; // [rsp+50h] [rbp-50h]
  __int64 v46; // [rsp+60h] [rbp-40h]

  v8 = a4;
  if ( !a5 )
  {
    v27 = *(_QWORD **)(a2 + 16);
    if ( v27 )
    {
      v28 = &a3[a4];
      while ( 1 )
      {
        v29 = sub_CF8CE0(v27);
        v30 = v29;
        if ( v29 )
        {
          sub_CF90E0((__int64)&v43, v27[3], v29);
          if ( v43.m128i_i32[0] )
          {
            if ( v28 != sub_CF8D60(a3, (__int64)v28, v43.m128i_i32) )
            {
              v34 = _mm_loadu_si128(&v43);
              v35 = v27[3];
              v46 = v44;
              v45 = v34;
              if ( a7(a8, v35, v30, v31, v32, v33, v34.m128i_i64[0], v34.m128i_i64[1], v44) )
                goto LABEL_19;
            }
          }
        }
        v27 = (_QWORD *)v27[1];
        if ( !v27 )
          goto LABEL_27;
      }
    }
    goto LABEL_27;
  }
  if ( *(_BYTE *)(a5 + 192) )
  {
    v11 = *(unsigned int *)(a5 + 184);
    if ( !(_DWORD)v11 )
      goto LABEL_27;
  }
  else
  {
    sub_CFDFC0(a5);
    v11 = *(unsigned int *)(a5 + 184);
    v8 = a4;
    if ( !(_DWORD)v11 )
      goto LABEL_27;
  }
  v12 = *(_QWORD *)(a5 + 168);
  v13 = (v11 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v14 = v12 + 88LL * v13;
  v15 = *(_QWORD *)(v14 + 24);
  if ( a2 != v15 )
  {
    v37 = 1;
    while ( v15 != -4096 )
    {
      v38 = v37 + 1;
      v13 = (v11 - 1) & (v37 + v13);
      v14 = v12 + 88LL * v13;
      v15 = *(_QWORD *)(v14 + 24);
      if ( a2 == v15 )
        goto LABEL_5;
      v37 = v38;
    }
    goto LABEL_27;
  }
LABEL_5:
  if ( v14 == v12 + 88 * v11
    || (v16 = *(_QWORD *)(v14 + 40), v17 = v16 + 32LL * *(unsigned int *)(v14 + 48), v17 == v16) )
  {
LABEL_27:
    *(_DWORD *)a1 = 0;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    return a1;
  }
  v39 = &a3[v8];
  while ( 1 )
  {
    v18 = *(_QWORD *)(v16 + 16);
    if ( v18 )
    {
      v19 = *(unsigned int *)(v16 + 24);
      if ( (_DWORD)v19 != -1 )
      {
        v20 = 0;
        if ( *(char *)(v18 + 7) < 0 )
        {
          v20 = sub_BD2BC0(*(_QWORD *)(v16 + 16));
          v19 = *(unsigned int *)(v16 + 24);
        }
        sub_CF90E0((__int64)&v43, v18, v20 + 16 * v19);
        if ( v43.m128i_i32[0] && v44 == a2 && v39 != sub_CF8D60(a3, (__int64)v39, v43.m128i_i32) )
        {
          v24 = 0;
          if ( *(char *)(v18 + 7) < 0 )
            v24 = sub_BD2BC0(v18);
          v25 = 16LL * *(unsigned int *)(v16 + 24);
          v45 = _mm_loadu_si128(&v43);
          v46 = v44;
          if ( a7(a8, v18, v24 + v25, v21, v22, v23, v45.m128i_i64[0], v45.m128i_i64[1], v44) )
            break;
        }
      }
    }
    v16 += 32;
    if ( v17 == v16 )
      goto LABEL_27;
  }
LABEL_19:
  v26 = v44;
  *(__m128i *)a1 = _mm_loadu_si128(&v43);
  *(_QWORD *)(a1 + 16) = v26;
  return a1;
}
