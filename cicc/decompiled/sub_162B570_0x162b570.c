// Function: sub_162B570
// Address: 0x162b570
//
__int64 __fastcall sub_162B570(
        __int64 a1,
        __int64 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // r12
  int v14; // r13d
  __int64 v15; // rdx
  int v16; // eax
  unsigned int v17; // r9d
  __int64 *v18; // rdx
  __int64 v19; // rdi
  unsigned int v20; // r8d
  int v21; // esi
  __int64 *v22; // rcx
  __int64 v23; // rdi
  int v24; // eax
  __int64 result; // rax
  __int64 v26; // r13
  __int64 v27; // r12
  __int64 v28; // rdi
  char v29; // dl
  __m128i v30; // xmm1
  int v31; // eax
  int v32; // r12d
  __int64 v33; // rdx
  __int64 *v34; // rsi
  int v35; // r8d
  __int64 *v36; // r9
  int v37; // eax
  char v38; // r10
  bool v39; // di
  int v40; // [rsp+Ch] [rbp-B4h]
  __int64 v41; // [rsp+28h] [rbp-98h] BYREF
  int v42; // [rsp+3Ch] [rbp-84h] BYREF
  __int64 v43; // [rsp+40h] [rbp-80h] BYREF
  __int64 v44; // [rsp+48h] [rbp-78h] BYREF
  __int64 *v45; // [rsp+50h] [rbp-70h] BYREF
  __int64 v46; // [rsp+58h] [rbp-68h] BYREF
  __m128 v47; // [rsp+60h] [rbp-60h]
  char v48; // [rsp+70h] [rbp-50h]
  __int64 v49; // [rsp+78h] [rbp-48h]
  char v50; // [rsp+80h] [rbp-40h]

  v47 = a10;
  v11 = *(unsigned int *)(a1 + 8);
  v41 = a1;
  v45 = *(__int64 **)(a1 - 8 * v11);
  v46 = *(_QWORD *)(a1 + 8 * (1 - v11));
  v12 = *(_BYTE *)(a1 + 40);
  v48 = v12;
  v13 = *(_QWORD *)(a2 + 8);
  v14 = *(_DWORD *)(a2 + 24);
  v50 = *(_BYTE *)(a1 + 56);
  if ( v50 )
  {
    v15 = *(_QWORD *)(a1 + 48);
    v49 = v15;
    if ( v14 )
      goto LABEL_3;
LABEL_12:
    ++*(_QWORD *)a2;
    v20 = 0;
    goto LABEL_13;
  }
  if ( !v14 )
    goto LABEL_12;
  v15 = 0;
LABEL_3:
  v44 = v15;
  if ( v12 )
  {
    v43 = v47.m128_i64[1];
    v16 = v47.m128_i32[0];
  }
  else
  {
    v43 = 0;
    v16 = 0;
  }
  v42 = v16;
  v40 = 1;
  v17 = (v14 - 1) & sub_15B5960((__int64 *)&v45, &v46, &v42, &v43, &v44);
  v18 = (__int64 *)(v13 + 8LL * v17);
  v19 = *v18;
  if ( *v18 == -8 )
  {
LABEL_21:
    v26 = *(_QWORD *)(a2 + 8);
    LODWORD(v27) = *(_DWORD *)(a2 + 24);
    goto LABEL_22;
  }
  while ( 1 )
  {
    if ( v19 != -16
      && v45 == *(__int64 **)(v19 - 8LL * *(unsigned int *)(v19 + 8))
      && v46 == *(_QWORD *)(v19 + 8 * (1LL - *(unsigned int *)(v19 + 8))) )
    {
      if ( *(_BYTE *)(v19 + 40) )
      {
        if ( !v48 || v47.m128_i32[0] != *(_DWORD *)(v19 + 24) || *(_QWORD *)(v19 + 32) != v47.m128_u64[1] )
          goto LABEL_9;
      }
      else if ( v48 )
      {
        goto LABEL_9;
      }
      v38 = *(_BYTE *)(v19 + 56);
      if ( v38 && v50 )
        v39 = *(_QWORD *)(v19 + 48) == v49;
      else
        v39 = v38 == v50;
      if ( v39 )
        break;
    }
LABEL_9:
    v17 = (v14 - 1) & (v40 + v17);
    v18 = (__int64 *)(v13 + 8LL * v17);
    v19 = *v18;
    if ( *v18 == -8 )
      goto LABEL_21;
    ++v40;
  }
  v26 = *(_QWORD *)(a2 + 8);
  v27 = *(unsigned int *)(a2 + 24);
  if ( v18 != (__int64 *)(v26 + 8 * v27) )
  {
    result = *v18;
    if ( *v18 )
      return result;
  }
LABEL_22:
  if ( !(_DWORD)v27 )
    goto LABEL_12;
  v28 = *(unsigned int *)(v41 + 8);
  v45 = *(__int64 **)(v41 - 8 * v28);
  v46 = *(_QWORD *)(v41 + 8 * (1 - v28));
  v48 = *(_BYTE *)(v41 + 40);
  v29 = *(_BYTE *)(v41 + 56);
  if ( v48 )
  {
    v30 = _mm_loadu_si128((const __m128i *)(v41 + 24));
    v50 = *(_BYTE *)(v41 + 56);
    v47 = (__m128)v30;
    if ( v29 )
    {
      v49 = *(_QWORD *)(v41 + 48);
      v44 = v49;
    }
    else
    {
      v44 = 0;
    }
    v43 = v47.m128_i64[1];
    v31 = v47.m128_i32[0];
  }
  else
  {
    v50 = *(_BYTE *)(v41 + 56);
    if ( v29 )
    {
      v49 = *(_QWORD *)(v41 + 48);
      v44 = v49;
    }
    else
    {
      v44 = 0;
    }
    v43 = 0;
    v31 = 0;
  }
  v42 = v31;
  v32 = v27 - 1;
  v23 = v41;
  LODWORD(v33) = v32 & sub_15B5960((__int64 *)&v45, &v46, &v42, &v43, &v44);
  v34 = (__int64 *)(v26 + 8LL * (unsigned int)v33);
  result = *v34;
  if ( v41 == *v34 )
    return result;
  v35 = 1;
  v22 = 0;
  if ( result == -8 )
  {
LABEL_36:
    v37 = *(_DWORD *)(a2 + 16);
    v20 = *(_DWORD *)(a2 + 24);
    if ( !v22 )
      v22 = v34;
    ++*(_QWORD *)a2;
    v24 = v37 + 1;
    if ( 4 * v24 < 3 * v20 )
    {
      if ( v20 - (v24 + *(_DWORD *)(a2 + 20)) > v20 >> 3 )
        goto LABEL_15;
      v21 = v20;
LABEL_14:
      sub_15BF140(a2, v21);
      sub_15B81B0(a2, &v41, &v45);
      v22 = v45;
      v23 = v41;
      v24 = *(_DWORD *)(a2 + 16) + 1;
LABEL_15:
      *(_DWORD *)(a2 + 16) = v24;
      if ( *v22 != -8 )
        --*(_DWORD *)(a2 + 20);
      *v22 = v23;
      return v41;
    }
LABEL_13:
    v21 = 2 * v20;
    goto LABEL_14;
  }
  while ( 1 )
  {
    if ( result != -16 || v22 )
      v34 = v22;
    v33 = v32 & (unsigned int)(v33 + v35);
    v36 = (__int64 *)(v26 + 8 * v33);
    result = *v36;
    if ( v41 == *v36 )
      return result;
    ++v35;
    v22 = v34;
    v34 = (__int64 *)(v26 + 8 * v33);
    if ( result == -8 )
      goto LABEL_36;
  }
}
