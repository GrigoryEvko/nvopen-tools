// Function: sub_3279310
// Address: 0x3279310
//
__int64 __fastcall sub_3279310(_QWORD *a1, __int64 a2)
{
  const __m128i *v4; // rax
  __int64 v5; // rcx
  unsigned __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 result; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  int v12; // esi
  __m128i si128; // xmm1
  __int64 v14; // r14
  unsigned __int16 v15; // ax
  __int64 v16; // r10
  int v17; // r14d
  __int64 *v18; // r11
  __int64 v19; // rax
  int v20; // r10d
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  int v24; // eax
  unsigned int v25; // eax
  __int64 v26; // r9
  unsigned __int64 v27; // rcx
  unsigned __int64 v28; // rax
  unsigned int v29; // edx
  unsigned int v30; // eax
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rcx
  unsigned int v33; // r15d
  unsigned __int64 v34; // rax
  bool v35; // r15
  __int64 v36; // rdi
  int v37; // eax
  int v38; // eax
  unsigned int v39; // esi
  __int128 v40; // [rsp-20h] [rbp-D0h]
  __int128 v41; // [rsp-10h] [rbp-C0h]
  int v42; // [rsp+10h] [rbp-A0h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  unsigned __int64 v44; // [rsp+18h] [rbp-98h]
  unsigned __int16 v45; // [rsp+28h] [rbp-88h]
  __int64 v46; // [rsp+28h] [rbp-88h]
  unsigned __int64 v47; // [rsp+28h] [rbp-88h]
  unsigned int v48; // [rsp+28h] [rbp-88h]
  unsigned int v49; // [rsp+28h] [rbp-88h]
  int v50; // [rsp+30h] [rbp-80h]
  __int64 v51; // [rsp+30h] [rbp-80h]
  unsigned int v52; // [rsp+30h] [rbp-80h]
  int v53; // [rsp+30h] [rbp-80h]
  int v54; // [rsp+30h] [rbp-80h]
  __int64 v55; // [rsp+38h] [rbp-78h]
  int v56; // [rsp+38h] [rbp-78h]
  __int64 v57; // [rsp+38h] [rbp-78h]
  int v58; // [rsp+38h] [rbp-78h]
  int v59; // [rsp+38h] [rbp-78h]
  __int64 v60; // [rsp+38h] [rbp-78h]
  __int128 v61; // [rsp+40h] [rbp-70h] BYREF
  __int64 v62; // [rsp+50h] [rbp-60h] BYREF
  int v63; // [rsp+58h] [rbp-58h]
  __m128i v64; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v65; // [rsp+70h] [rbp-40h]
  __int64 v66; // [rsp+78h] [rbp-38h]

  v4 = *(const __m128i **)(a2 + 40);
  v5 = v4->m128i_i64[0];
  v6 = v4[2].m128i_u64[1];
  v7 = v4[3].m128i_i64[0];
  v61 = (__int128)_mm_loadu_si128(v4);
  v55 = v5;
  v8 = v4->m128i_u32[2];
  result = sub_3401190(*a1, v61, *((_QWORD *)&v61 + 1), v6, v7);
  if ( !result )
  {
    v10 = *(_QWORD *)(a2 + 80);
    v62 = v10;
    if ( v10 )
      sub_B96E90((__int64)&v62, v10, 1);
    v11 = *a1;
    v12 = *(_DWORD *)(a2 + 24);
    si128 = _mm_load_si128((const __m128i *)&v61);
    v63 = *(_DWORD *)(a2 + 72);
    v14 = *(_QWORD *)(v55 + 48) + 16 * v8;
    v15 = *(_WORD *)v14;
    v16 = *(_QWORD *)(v14 + 8);
    v65 = v6;
    v45 = v15;
    v17 = v15;
    v50 = v16;
    v66 = v7;
    v64 = si128;
    result = sub_3402EA0(v11, v12, (unsigned int)&v62, v15, v16, 0, (__int64)&v64, 2);
    v18 = &v62;
    if ( result )
      goto LABEL_6;
    v56 = v50;
    v19 = sub_33DFBC0(v6, v7, 0, 0);
    v20 = v50;
    v18 = &v62;
    v21 = v19;
    if ( *((_BYTE *)a1 + 33) )
    {
      v22 = a1[1];
      v23 = 1;
      if ( v45 != 1 )
      {
        if ( !v45 )
          goto LABEL_15;
        v23 = v45;
        if ( !*(_QWORD *)(v22 + 8LL * v45 + 112) )
          goto LABEL_15;
      }
      if ( (*(_BYTE *)(v22 + 500 * v23 + 6604) & 0xFB) != 0 )
        goto LABEL_15;
    }
    v24 = *(_DWORD *)(a2 + 24);
    if ( v24 != 86 )
    {
      if ( v24 != 87 || !v21 )
        goto LABEL_15;
LABEL_21:
      v53 = v20;
      v57 = *(_QWORD *)(v21 + 96);
      sub_33DD090(&v64, *a1, v61, *((_QWORD *)&v61 + 1), 0);
      v29 = v64.m128i_u32[2];
      v20 = v53;
      v18 = &v62;
      if ( v64.m128i_i32[2] > 0x40u )
      {
        v48 = v64.m128i_u32[2];
        v30 = sub_C44500((__int64)&v64);
        v20 = v53;
        v29 = v48;
        v18 = &v62;
      }
      else if ( v64.m128i_i32[2] )
      {
        v30 = 64;
        if ( v64.m128i_i64[0] << (64 - v64.m128i_i8[8]) != -1 )
        {
          _BitScanReverse64(&v31, ~(v64.m128i_i64[0] << (64 - v64.m128i_i8[8])));
          v30 = v31 ^ 0x3F;
        }
      }
      else
      {
        v30 = 0;
      }
      v32 = v30;
      v33 = *(_DWORD *)(v57 + 32);
      if ( v33 > 0x40 )
      {
        v49 = v29;
        v54 = v20;
        v44 = v30;
        v38 = sub_C444A0(v57 + 24);
        v39 = v33;
        v35 = 0;
        v20 = v54;
        v29 = v49;
        v18 = &v62;
        if ( v39 - v38 > 0x40 )
          goto LABEL_28;
        v32 = v44;
        v34 = **(_QWORD **)(v57 + 24);
      }
      else
      {
        v34 = *(_QWORD *)(v57 + 24);
      }
      v35 = v32 >= v34;
LABEL_28:
      if ( (unsigned int)v66 > 0x40 && v65 )
      {
        v58 = v20;
        j_j___libc_free_0_0(v65);
        v29 = v64.m128i_u32[2];
        v18 = &v62;
        v20 = v58;
      }
      if ( v29 > 0x40 && v64.m128i_i64[0] )
      {
        v59 = v20;
        j_j___libc_free_0_0(v64.m128i_u64[0]);
        v18 = &v62;
        v20 = v59;
      }
      if ( !v35 )
        goto LABEL_15;
LABEL_35:
      *((_QWORD *)&v41 + 1) = v7;
      v36 = *a1;
      *(_QWORD *)&v41 = v6;
      v40 = v61;
      *(_QWORD *)&v61 = &v62;
      result = sub_3406EB0(v36, 190, (unsigned int)&v62, v17, v20, v26, v40, v41);
      v18 = (__int64 *)v61;
LABEL_6:
      if ( v62 )
      {
        *(_QWORD *)&v61 = result;
        sub_B91220((__int64)v18, v62);
        return v61;
      }
      return result;
    }
    if ( !v21 )
    {
LABEL_15:
      result = 0;
      goto LABEL_6;
    }
    v46 = v21;
    v51 = *(_QWORD *)(v21 + 96);
    v25 = sub_33D4D80(*a1, v61, *((_QWORD *)&v61 + 1), 0);
    v26 = v51;
    v21 = v46;
    v18 = &v62;
    v20 = v56;
    v27 = v25;
    v52 = *(_DWORD *)(v51 + 32);
    if ( v52 > 0x40 )
    {
      v42 = v56;
      v43 = v46;
      v47 = v25;
      v60 = v26;
      v37 = sub_C444A0(v26 + 24);
      LODWORD(v26) = v60;
      v27 = v47;
      v21 = v43;
      v20 = v42;
      v18 = &v62;
      if ( v52 - v37 > 0x40 )
        goto LABEL_20;
      v28 = **(_QWORD **)(v60 + 24);
    }
    else
    {
      v28 = *(_QWORD *)(v26 + 24);
    }
    if ( v27 > v28 )
      goto LABEL_35;
LABEL_20:
    if ( *(_DWORD *)(a2 + 24) == 87 )
      goto LABEL_21;
    goto LABEL_15;
  }
  return result;
}
