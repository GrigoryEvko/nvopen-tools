// Function: sub_16FACE0
// Address: 0x16face0
//
__int64 __fastcall sub_16FACE0(__int64 a1)
{
  __int64 result; // rax
  __int64 v3; // rax
  int v4; // edx
  unsigned int v5; // r12d
  unsigned int v6; // esi
  int v7; // r15d
  int v8; // r8d
  int v9; // r9d
  int v10; // edx
  _BYTE *v11; // rax
  __int64 v12; // rdi
  size_t v13; // r15
  int v14; // edi
  const void *v15; // r15
  unsigned int v16; // r8d
  _BYTE *v17; // r14
  __int64 v18; // rdi
  size_t v19; // r11
  int v20; // r15d
  size_t v21; // r12
  size_t v22; // rbx
  _BYTE *v23; // r10
  _BYTE *v24; // rdi
  int v25; // eax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rdi
  __int64 *v31; // rax
  __m128i v32; // xmm0
  _BYTE *v33; // rsi
  __int64 *v34; // r12
  size_t v35; // rdx
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  size_t v38; // rdx
  __int64 v39; // rax
  _QWORD *v40; // rdi
  __int64 v41; // [rsp+8h] [rbp-1E8h]
  size_t v42; // [rsp+10h] [rbp-1E0h]
  _BYTE *v43; // [rsp+10h] [rbp-1E0h]
  unsigned __int8 v44; // [rsp+1Fh] [rbp-1D1h]
  unsigned int src; // [rsp+20h] [rbp-1D0h]
  unsigned int srca; // [rsp+20h] [rbp-1D0h]
  _BYTE *srcb; // [rsp+20h] [rbp-1D0h]
  unsigned __int8 v48; // [rsp+28h] [rbp-1C8h]
  char v49; // [rsp+3Eh] [rbp-1B2h] BYREF
  unsigned __int8 v50; // [rsp+3Fh] [rbp-1B1h] BYREF
  unsigned int v51; // [rsp+40h] [rbp-1B0h] BYREF
  _DWORD n[3]; // [rsp+44h] [rbp-1ACh] BYREF
  _QWORD *v53; // [rsp+50h] [rbp-1A0h] BYREF
  size_t v54; // [rsp+58h] [rbp-198h]
  _QWORD v55[2]; // [rsp+60h] [rbp-190h] BYREF
  int v56; // [rsp+70h] [rbp-180h]
  __m128i v57; // [rsp+78h] [rbp-178h] BYREF
  void *dest; // [rsp+88h] [rbp-168h]
  size_t v59; // [rsp+90h] [rbp-160h]
  _QWORD v60[3]; // [rsp+98h] [rbp-158h] BYREF
  _BYTE *v61; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-138h]
  _BYTE v63[304]; // [rsp+C0h] [rbp-130h] BYREF

  sub_16F7930(a1, 1u);
  v50 = 0;
  v44 = sub_16FA830(a1, &v49, &v51, &v50);
  if ( !v44 )
    return 0;
  result = v50;
  if ( v50 )
    return result;
  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_DWORD *)(a1 + 56);
  v5 = 0;
  n[0] = 0;
  v6 = v51;
  v41 = v3;
  if ( v4 >= 0 )
    v5 = *(_DWORD *)(a1 + 56);
  v7 = v51;
  if ( v51 )
  {
    v61 = v63;
    v62 = 0x10000000000LL;
    goto LABEL_18;
  }
  if ( !(unsigned __int8)sub_16F7D90(a1, &v51, v5, n, &v50) )
    return 0;
  v61 = v63;
  v62 = 0x10000000000LL;
  if ( !v50 )
  {
    v6 = v51;
LABEL_18:
    while ( 1 )
    {
      result = sub_16F7ED0(a1, v6, v5, &v50);
      if ( !(_BYTE)result )
        goto LABEL_49;
      v17 = *(_BYTE **)(a1 + 40);
      v15 = v17;
      if ( v50 )
      {
        v21 = (unsigned int)v62;
        v16 = n[0];
        v7 = v62;
        goto LABEL_26;
      }
      sub_16F77C0(a1, (char *)sub_16F6380, 0);
      if ( *(_BYTE **)(a1 + 40) != v17 )
      {
        v18 = (unsigned int)v62;
        v19 = n[0];
        v10 = v62;
        v20 = n[0];
        if ( n[0] > HIDWORD(v62) - (unsigned __int64)(unsigned int)v62 )
        {
          v42 = n[0];
          sub_16CD150((__int64)&v61, v63, n[0] + (unsigned __int64)(unsigned int)v62, 1, v8, v9);
          v18 = (unsigned int)v62;
          v19 = v42;
          v10 = v62;
        }
        if ( v19 )
        {
          memset(&v61[v18], 10, v19);
          v10 = v62;
        }
        v11 = *(_BYTE **)(a1 + 40);
        v12 = (unsigned int)(v20 + v10);
        LODWORD(v62) = v20 + v10;
        v13 = v11 - v17;
        if ( v11 - v17 > (unsigned __int64)HIDWORD(v62) - v12 )
        {
          v43 = v11;
          sub_16CD150((__int64)&v61, v63, v13 + v12, 1, v8, v9);
          v12 = (unsigned int)v62;
          v11 = v43;
        }
        if ( v11 != v17 )
        {
          memcpy(&v61[v12], v17, v13);
          LODWORD(v12) = v62;
        }
        n[0] = 0;
        v14 = v13 + v12;
        v15 = *(const void **)(a1 + 40);
        LODWORD(v62) = v14;
      }
      if ( *(const void **)(a1 + 48) == v15 )
      {
        v21 = (unsigned int)v62;
        v16 = n[0];
        v7 = v62;
        goto LABEL_58;
      }
      if ( !(unsigned __int8)sub_16F7970(a1) )
      {
        v21 = (unsigned int)v62;
        v16 = n[0];
        v17 = *(_BYTE **)(a1 + 40);
        v7 = v62;
        goto LABEL_26;
      }
      v16 = ++n[0];
      if ( v50 )
      {
        v21 = (unsigned int)v62;
        v17 = *(_BYTE **)(a1 + 40);
        v7 = v62;
        goto LABEL_26;
      }
      v6 = v51;
    }
  }
  v16 = n[0];
  v17 = *(_BYTE **)(a1 + 40);
  v21 = 0;
LABEL_26:
  if ( v17 == *(_BYTE **)(a1 + 48) )
  {
LABEL_58:
    if ( !v16 )
    {
      n[0] = 1;
      v16 = 1;
    }
  }
  if ( v49 != 45 )
  {
    if ( v49 == 43 )
    {
      v22 = v16;
      if ( v16 <= HIDWORD(v62) - v21 )
      {
LABEL_32:
        v23 = v61;
        v24 = &v61[v21];
        v21 = v7 + v16;
        v7 += v16;
        if ( !v22 )
          goto LABEL_34;
        goto LABEL_33;
      }
LABEL_31:
      src = v16;
      sub_16CD150((__int64)&v61, v63, v22 + v21, 1, v16, v9);
      v21 = (unsigned int)v62;
      v16 = src;
      v7 = v62;
      goto LABEL_32;
    }
    if ( v7 )
    {
      v16 = 1;
      v22 = 1;
      if ( HIDWORD(v62) != v21 )
      {
        v24 = &v61[v21];
LABEL_33:
        srca = v16;
        memset(v24, 10, v22);
        v21 = (unsigned int)v62 + srca;
        v23 = v61;
        v7 = v62 + srca;
        goto LABEL_34;
      }
      goto LABEL_31;
    }
  }
  v23 = v61;
LABEL_34:
  v25 = *(_DWORD *)(a1 + 68);
  LODWORD(v62) = v7;
  if ( !v25 )
    *(_BYTE *)(a1 + 73) = 1;
  v26 = *(_QWORD *)(a1 + 40);
  v59 = 0;
  dest = v60;
  LOBYTE(v60[0]) = 0;
  v56 = 19;
  v57.m128i_i64[0] = v41;
  v57.m128i_i64[1] = v26 - v41;
  if ( !v23 )
  {
    LOBYTE(v55[0]) = 0;
    v38 = 0;
    v28 = v60;
    v53 = v55;
LABEL_56:
    v59 = v38;
    *((_BYTE *)v28 + v38) = 0;
    v29 = v53;
    goto LABEL_44;
  }
  *(_QWORD *)&n[1] = v21;
  v53 = v55;
  if ( v21 > 0xF )
  {
    srcb = v23;
    v39 = sub_22409D0(&v53, &n[1], 0);
    v23 = srcb;
    v53 = (_QWORD *)v39;
    v40 = (_QWORD *)v39;
    v55[0] = *(_QWORD *)&n[1];
  }
  else
  {
    if ( v21 == 1 )
    {
      LOBYTE(v55[0]) = *v23;
      v27 = v55;
      goto LABEL_40;
    }
    if ( !v21 )
    {
      v27 = v55;
      goto LABEL_40;
    }
    v40 = v55;
  }
  memcpy(v40, v23, v21);
  v21 = *(_QWORD *)&n[1];
  v27 = v53;
LABEL_40:
  v54 = v21;
  *((_BYTE *)v27 + v21) = 0;
  v28 = dest;
  v29 = dest;
  if ( v53 == v55 )
  {
    v38 = v54;
    if ( v54 )
    {
      if ( v54 == 1 )
        *(_BYTE *)dest = v55[0];
      else
        memcpy(dest, v55, v54);
      v38 = v54;
      v28 = dest;
    }
    goto LABEL_56;
  }
  if ( dest == v60 )
  {
    dest = v53;
    v59 = v54;
    v60[0] = v55[0];
    goto LABEL_63;
  }
  v30 = v60[0];
  dest = v53;
  v59 = v54;
  v60[0] = v55[0];
  if ( !v29 )
  {
LABEL_63:
    v53 = v55;
    v29 = v55;
    goto LABEL_44;
  }
  v53 = v29;
  v55[0] = v30;
LABEL_44:
  v54 = 0;
  *v29 = 0;
  if ( v53 != v55 )
    j_j___libc_free_0(v53, v55[0] + 1LL);
  v31 = (__int64 *)sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v32 = _mm_loadu_si128(&v57);
  v33 = dest;
  v34 = v31;
  *v31 = 0;
  v35 = v59;
  v31[1] = 0;
  LODWORD(v31) = v56;
  *(__m128i *)(v34 + 3) = v32;
  *((_DWORD *)v34 + 4) = (_DWORD)v31;
  v34[5] = (__int64)(v34 + 7);
  sub_16F6740(v34 + 5, v33, (__int64)&v33[v35]);
  v36 = *(_QWORD *)(a1 + 184);
  v34[1] = a1 + 184;
  v36 &= 0xFFFFFFFFFFFFFFF8LL;
  *v34 = v36 | *v34 & 7;
  *(_QWORD *)(v36 + 8) = v34;
  v37 = dest;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | (unsigned __int64)v34;
  if ( v37 != v60 )
    j_j___libc_free_0(v37, v60[0] + 1LL);
  result = v44;
LABEL_49:
  if ( v61 != v63 )
  {
    v48 = result;
    _libc_free((unsigned __int64)v61);
    return v48;
  }
  return result;
}
