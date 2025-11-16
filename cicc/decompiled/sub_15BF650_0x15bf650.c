// Function: sub_15BF650
// Address: 0x15bf650
//
__int64 __fastcall sub_15BF650(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        __int64 a5,
        unsigned int a6,
        char a7)
{
  __int64 v7; // rax
  char v13; // dl
  __int8 v14; // cl
  __int64 v15; // r9
  int v16; // r10d
  int v17; // eax
  int v18; // eax
  int v19; // r10d
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 result; // rax
  __int8 v24; // r10
  char v25; // r15
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rbx
  __int64 v31; // rdx
  __int64 v32; // rax
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  char v35; // cl
  bool v36; // dl
  __int64 v37; // [rsp+8h] [rbp-D8h]
  __int64 v38; // [rsp+10h] [rbp-D0h]
  __int64 v39; // [rsp+18h] [rbp-C8h]
  __int64 v40; // [rsp+20h] [rbp-C0h]
  int v41; // [rsp+28h] [rbp-B8h]
  int i; // [rsp+28h] [rbp-B8h]
  __int8 v43; // [rsp+2Ch] [rbp-B4h]
  int v44; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 v45; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v46; // [rsp+48h] [rbp-98h] BYREF
  __m128i v47[2]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v48; // [rsp+70h] [rbp-70h] BYREF
  __int64 v49; // [rsp+78h] [rbp-68h] BYREF
  __m128i v50; // [rsp+80h] [rbp-60h]
  char v51; // [rsp+90h] [rbp-50h]
  __int64 v52; // [rsp+98h] [rbp-48h]
  char v53; // [rsp+A0h] [rbp-40h]

  if ( a6 )
    goto LABEL_18;
  v13 = *(_BYTE *)(a5 + 8);
  if ( v13 )
    v7 = *(_QWORD *)a5;
  v14 = a4[1].m128i_i8[0];
  v48 = a2;
  v49 = a3;
  if ( v14 )
  {
    v51 = 1;
    v47[0] = _mm_loadu_si128(a4);
    v50 = v47[0];
  }
  else
  {
    v51 = 0;
  }
  v15 = *a1;
  v53 = v13;
  v16 = *(_DWORD *)(v15 + 872);
  v39 = *(_QWORD *)(v15 + 856);
  if ( v13 )
  {
    v52 = v7;
    if ( !v16 )
      goto LABEL_17;
  }
  else
  {
    v7 = 0;
    if ( !v16 )
      goto LABEL_17;
  }
  v46 = v7;
  if ( v14 )
  {
    v45 = v50.m128i_i64[1];
    v17 = v50.m128i_i32[0];
  }
  else
  {
    v45 = 0;
    v17 = 0;
  }
  v38 = a3;
  v41 = v16;
  v37 = v15;
  v44 = v17;
  v18 = sub_15B5960(&v48, &v49, &v44, &v45, &v46);
  a3 = v38;
  v19 = v41 - 1;
  v20 = (v41 - 1) & v18;
  v21 = (__int64 *)(v39 + 8LL * v20);
  v22 = *v21;
  if ( *v21 == -8 )
    goto LABEL_17;
  for ( i = 1; ; ++i )
  {
    if ( v22 != -16
      && v48 == *(_QWORD *)(v22 - 8LL * *(unsigned int *)(v22 + 8))
      && v49 == *(_QWORD *)(v22 + 8 * (1LL - *(unsigned int *)(v22 + 8))) )
    {
      if ( *(_BYTE *)(v22 + 40) )
      {
        if ( !v51 || *(_DWORD *)(v22 + 24) != v50.m128i_i32[0] || *(_QWORD *)(v22 + 32) != v50.m128i_i64[1] )
          goto LABEL_14;
      }
      else if ( v51 )
      {
        goto LABEL_14;
      }
      v35 = *(_BYTE *)(v22 + 56);
      if ( v35 && v53 )
        v36 = *(_QWORD *)(v22 + 48) == v52;
      else
        v36 = v35 == v53;
      if ( v36 )
        break;
    }
LABEL_14:
    v20 = v19 & (i + v20);
    v21 = (__int64 *)(v39 + 8LL * v20);
    v22 = *v21;
    if ( *v21 == -8 )
      goto LABEL_17;
  }
  if ( v21 != (__int64 *)(*(_QWORD *)(v37 + 856) + 8LL * *(unsigned int *)(v37 + 872)) )
  {
    result = *v21;
    if ( result )
      return result;
  }
LABEL_17:
  result = 0;
  if ( !a7 )
    return result;
LABEL_18:
  v24 = a4[1].m128i_i8[0];
  v48 = a2;
  v49 = a3;
  if ( v24 )
  {
    v25 = *(_BYTE *)(a5 + 8);
    v31 = *a1;
    v50.m128i_i64[0] = a4->m128i_i64[1];
    if ( v25 )
    {
      v32 = *(_QWORD *)a5;
      v33 = _mm_loadu_si128(a4);
      v28 = v31 + 848;
      v50.m128i_i64[1] = v32;
      v40 = v32;
      v47[0] = v33;
LABEL_21:
      v25 = 1;
    }
    else
    {
      v34 = _mm_loadu_si128(a4);
      v50.m128i_i64[1] = 0;
      v28 = v31 + 848;
      v47[0] = v34;
    }
  }
  else
  {
    v25 = *(_BYTE *)(a5 + 8);
    v26 = *a1;
    v50.m128i_i64[0] = 0;
    if ( v25 )
    {
      v27 = *(_QWORD *)a5;
      v28 = v26 + 848;
      v40 = v27;
      v50.m128i_i64[1] = v27;
      goto LABEL_21;
    }
    v50.m128i_i64[1] = 0;
    v28 = v26 + 848;
  }
  v43 = v24;
  v29 = sub_161E980(64, 4);
  v30 = v29;
  if ( v29 )
  {
    sub_1623D80(v29, (_DWORD)a1, 15, a6, (unsigned int)&v48, 4, 0, 0);
    *(_WORD *)(v30 + 2) = 41;
    *(_BYTE *)(v30 + 40) = v43;
    if ( v43 )
      *(__m128i *)(v30 + 24) = _mm_loadu_si128(v47);
    *(_BYTE *)(v30 + 56) = v25;
    if ( v25 )
      *(_QWORD *)(v30 + 48) = v40;
  }
  return sub_15BF400(v30, a6, v28);
}
