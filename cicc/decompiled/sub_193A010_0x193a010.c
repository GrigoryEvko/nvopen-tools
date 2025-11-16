// Function: sub_193A010
// Address: 0x193a010
//
__int64 __fastcall sub_193A010(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rax
  __int64 *v7; // rsi
  unsigned int v8; // r15d
  char v9; // dl
  char v11; // al
  __int64 v12; // rdx
  __int64 *v13; // rcx
  __int64 v14; // r13
  int v15; // eax
  __int64 ***v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r8
  int v22; // r9d
  char v23; // al
  __int16 v24; // ax
  __int64 v25; // rax
  __m128i v26; // xmm1
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // r14
  __int64 v30; // rax
  __int64 *v31; // r14
  __int64 v32; // r14
  __int64 v33; // r8
  __int64 v34; // rax
  bool v35; // bl
  unsigned int v36; // edx
  bool v37; // al
  __int64 v38; // [rsp+8h] [rbp-108h]
  __int64 *v39; // [rsp+10h] [rbp-100h]
  __int64 v40; // [rsp+18h] [rbp-F8h]
  __int64 *v41; // [rsp+20h] [rbp-F0h]
  __int64 v42; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v43; // [rsp+38h] [rbp-D8h]
  __int64 v44; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v45; // [rsp+48h] [rbp-C8h]
  __m128i v46; // [rsp+50h] [rbp-C0h] BYREF
  __m128i v47; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v48; // [rsp+70h] [rbp-A0h] BYREF
  _BYTE *v49; // [rsp+78h] [rbp-98h]
  _BYTE *v50; // [rsp+80h] [rbp-90h]
  __int64 v51; // [rsp+88h] [rbp-88h]
  int v52; // [rsp+90h] [rbp-80h]
  _BYTE v53[120]; // [rsp+98h] [rbp-78h] BYREF

  v6 = *(__int64 **)(a3 + 8);
  if ( *(__int64 **)(a3 + 16) != v6 )
  {
LABEL_2:
    v7 = (__int64 *)a1;
    v8 = 1;
    sub_16CCBA0(a3, a1);
    if ( !v9 )
      return v8;
    goto LABEL_6;
  }
  v12 = *(unsigned int *)(a3 + 28);
  v7 = &v6[v12];
  if ( v6 == v7 )
    goto LABEL_31;
  v13 = 0;
  do
  {
    if ( a1 == *v6 )
      return 1;
    if ( *v6 == -2 )
      v13 = v6;
    ++v6;
  }
  while ( v7 != v6 );
  if ( !v13 )
  {
LABEL_31:
    if ( (unsigned int)v12 >= *(_DWORD *)(a3 + 24) )
      goto LABEL_2;
    *(_DWORD *)(a3 + 28) = v12 + 1;
    *v7 = a1;
    ++*(_QWORD *)a3;
  }
  else
  {
    *v13 = a1;
    --*(_DWORD *)(a3 + 32);
    ++*(_QWORD *)a3;
  }
LABEL_6:
  v11 = *(_BYTE *)(a1 + 16);
  if ( v11 == 50 )
  {
    v28 = *(_QWORD *)(a1 - 48);
    if ( !v28 )
      return 0;
    v29 = *(_QWORD *)(a1 - 24);
    if ( !v29 )
      return 0;
    goto LABEL_35;
  }
  if ( v11 == 5 )
  {
    if ( *(_WORD *)(a1 + 18) != 26 )
      return 0;
    v28 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( !v28 )
      return 0;
    v29 = *(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
    if ( !v29 )
      return 0;
LABEL_35:
    v48 = 0;
    v49 = v53;
    v50 = v53;
    v51 = 8;
    v52 = 0;
    v8 = sub_193A010(v28, a2, &v48);
    if ( v50 != v49 )
      _libc_free((unsigned __int64)v50);
    if ( (_BYTE)v8 )
    {
      v49 = v53;
      v48 = 0;
      v50 = v53;
      v51 = 8;
      v52 = 0;
      v8 = sub_193A010(v29, a2, &v48);
      if ( v50 != v49 )
        _libc_free((unsigned __int64)v50);
    }
    return v8;
  }
  if ( v11 != 75 )
    return 0;
  v14 = *(_QWORD *)(a1 - 48);
  if ( *(_BYTE *)(*(_QWORD *)v14 + 8LL) != 11 )
    return 0;
  v15 = *(unsigned __int16 *)(a1 + 18);
  BYTE1(v15) &= ~0x80u;
  if ( ((v15 - 34) & 0xFFFFFFFD) != 0 )
    return 0;
  v16 = *(__int64 ****)(a1 - 24);
  if ( v15 == 34 )
  {
    v14 = *(_QWORD *)(a1 - 24);
    v16 = *(__int64 ****)(a1 - 48);
  }
  v17 = sub_15F2050(a1);
  v40 = sub_1632FA0(v17);
  v20 = sub_15A06D0(*v16, (__int64)v7, v18, v19);
  v47.m128i_i64[0] = (__int64)v16;
  v46.m128i_i64[0] = v14;
  v46.m128i_i64[1] = v20;
  v47.m128i_i64[1] = a1;
  v8 = sub_14C2730((__int64 *)v16, v40, 0, 0, 0, 0);
  if ( !(_BYTE)v8 )
    return 0;
  v39 = (__int64 *)sub_16498A0(a1);
  while ( 1 )
  {
    while ( 1 )
    {
      v23 = *(_BYTE *)(v14 + 16);
      if ( v23 == 35 )
      {
        if ( !*(_QWORD *)(v14 - 48) )
          goto LABEL_28;
        v30 = *(_QWORD *)(v14 - 24);
        if ( *(_BYTE *)(v30 + 16) != 13 )
          goto LABEL_28;
        v14 = *(_QWORD *)(v14 - 48);
        goto LABEL_46;
      }
      if ( v23 != 5 )
        break;
      v24 = *(_WORD *)(v14 + 18);
      if ( v24 != 11 )
      {
        if ( v24 != 27 )
          goto LABEL_28;
        v41 = *(__int64 **)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
        if ( !v41 )
          goto LABEL_28;
        v32 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
        if ( *(_BYTE *)(v32 + 16) != 13 )
          goto LABEL_28;
        goto LABEL_54;
      }
      if ( !*(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)) )
        goto LABEL_28;
      v30 = *(_QWORD *)(v14 + 24 * (1LL - (*(_DWORD *)(v14 + 20) & 0xFFFFFFF)));
      if ( *(_BYTE *)(v30 + 16) != 13 )
        goto LABEL_28;
      v14 = *(_QWORD *)(v14 - 24LL * (*(_DWORD *)(v14 + 20) & 0xFFFFFFF));
LABEL_46:
      v31 = (__int64 *)(v30 + 24);
      v46.m128i_i64[0] = v14;
      LODWORD(v49) = *(_DWORD *)(v46.m128i_i64[1] + 32);
      if ( (unsigned int)v49 > 0x40 )
        sub_16A4FD0((__int64)&v48, (const void **)(v46.m128i_i64[1] + 24));
      else
        v48 = *(_QWORD *)(v46.m128i_i64[1] + 24);
      sub_16A7200((__int64)&v48, v31);
      v45 = (unsigned int)v49;
      v44 = v48;
      v46.m128i_i64[1] = sub_159C0E0(v39, (__int64)&v44);
      if ( v45 > 0x40 && v44 )
        j_j___libc_free_0_0(v44);
    }
    if ( v23 != 51 )
      break;
    v41 = *(__int64 **)(v14 - 48);
    if ( !v41 )
      break;
    v32 = *(_QWORD *)(v14 - 24);
    if ( *(_BYTE *)(v32 + 16) != 13 )
      break;
LABEL_54:
    sub_14C2530((__int64)&v48, v41, v40, 0, 0, 0, 0, 0);
    v45 = *(_DWORD *)(v32 + 32);
    if ( v45 <= 0x40 )
    {
      v33 = *(_QWORD *)(v32 + 24);
      v34 = v33;
LABEL_56:
      v21 = v48 & v33;
LABEL_57:
      v35 = v34 == v21;
      goto LABEL_58;
    }
    sub_16A4FD0((__int64)&v44, (const void **)(v32 + 24));
    if ( v45 <= 0x40 )
    {
      v33 = v44;
      v34 = *(_QWORD *)(v32 + 24);
      goto LABEL_56;
    }
    sub_16A8890(&v44, &v48);
    v36 = v45;
    v21 = v44;
    v45 = 0;
    v43 = v36;
    v42 = v44;
    if ( v36 <= 0x40 )
    {
      v34 = *(_QWORD *)(v32 + 24);
      goto LABEL_57;
    }
    v38 = v44;
    v37 = sub_16A5220((__int64)&v42, (const void **)(v32 + 24));
    LODWORD(v21) = v38;
    v35 = v37;
    if ( v38 )
    {
      j_j___libc_free_0_0(v38);
      if ( v45 > 0x40 )
      {
        if ( v44 )
          j_j___libc_free_0_0(v44);
      }
    }
LABEL_58:
    if ( v35 )
    {
      v46.m128i_i64[0] = (__int64)v41;
      v45 = *(_DWORD *)(v46.m128i_i64[1] + 32);
      if ( v45 > 0x40 )
        sub_16A4FD0((__int64)&v44, (const void **)(v46.m128i_i64[1] + 24));
      else
        v44 = *(_QWORD *)(v46.m128i_i64[1] + 24);
      sub_16A7200((__int64)&v44, (__int64 *)(v32 + 24));
      v43 = v45;
      v42 = v44;
      v46.m128i_i64[1] = sub_159C0E0(v39, (__int64)&v42);
      if ( v43 > 0x40 )
      {
        if ( v42 )
          j_j___libc_free_0_0(v42);
      }
    }
    if ( (unsigned int)v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    if ( (unsigned int)v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    if ( !v35 )
      break;
    v14 = v46.m128i_i64[0];
  }
LABEL_28:
  v25 = *(unsigned int *)(a2 + 8);
  if ( (unsigned int)v25 >= *(_DWORD *)(a2 + 12) )
  {
    sub_16CD150(a2, (const void *)(a2 + 16), 0, 32, v21, v22);
    v25 = *(unsigned int *)(a2 + 8);
  }
  v26 = _mm_loadu_si128(&v47);
  v27 = *(_QWORD *)a2 + 32 * v25;
  *(__m128i *)v27 = _mm_loadu_si128(&v46);
  *(__m128i *)(v27 + 16) = v26;
  ++*(_DWORD *)(a2 + 8);
  return v8;
}
