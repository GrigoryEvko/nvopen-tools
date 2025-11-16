// Function: sub_103C0D0
// Address: 0x103c0d0
//
__int64 __fastcall sub_103C0D0(__int64 a1, __int64 *a2, __int64 ***a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int8 v8; // r13
  bool v11; // zf
  __int64 **v12; // rsi
  __int64 result; // rax
  __int64 **v14; // rdi
  unsigned __int8 v15; // dl
  char v16; // r13
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rdx
  __int64 **v24; // rcx
  const __m128i *v25; // rdx
  __m128i *v26; // rax
  __int64 v27; // rcx
  __int64 **v28; // rdx
  __int64 v29; // r8
  __int64 *v30; // r13
  __int64 *v31; // rdx
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // rdi
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // r8
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 **v48; // rcx
  const __m128i *v49; // rdx
  __m128i *v50; // rax
  unsigned __int8 v51; // dl
  __int64 **v52; // r12
  int v53; // edx
  int v54; // r8d
  __int64 **v55; // r12
  unsigned __int8 v56; // [rsp+1Bh] [rbp-365h]
  char v57; // [rsp+1Ch] [rbp-364h]
  char v58; // [rsp+1Ch] [rbp-364h]
  __int64 *v59; // [rsp+20h] [rbp-360h] BYREF
  __int64 v60; // [rsp+28h] [rbp-358h]
  __int64 v61; // [rsp+30h] [rbp-350h]
  __int64 **v62; // [rsp+40h] [rbp-340h] BYREF
  __int64 v63; // [rsp+48h] [rbp-338h]
  _BYTE v64[816]; // [rsp+50h] [rbp-330h] BYREF

  v6 = a4;
  v7 = (__int64)a3;
  v8 = a5;
  v11 = *(_BYTE *)(a4 + 28) == 0;
  v62 = (__int64 **)v64;
  v12 = (__int64 **)*a2;
  v56 = a5;
  v63 = 0x2000000000LL;
  if ( v11 )
  {
LABEL_11:
    v57 = a6;
    result = (__int64)sub_C8CC70(v6, (__int64)v12, (__int64)a3, a4, a5, a6);
    LOBYTE(a6) = v57;
    if ( ((v15 ^ 1) & v8) == 0 )
      goto LABEL_12;
LABEL_7:
    v14 = v62;
    if ( v62 == (__int64 **)v64 )
      return result;
    return _libc_free(v14, v12);
  }
  result = *(_QWORD *)(a4 + 8);
  a4 = *(unsigned int *)(a4 + 20);
  a3 = (__int64 ***)(result + 8 * a4);
  if ( (__int64 ***)result == a3 )
  {
LABEL_10:
    if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 16) )
      goto LABEL_11;
    *(_DWORD *)(v6 + 20) = a4 + 1;
    *a3 = v12;
    ++*(_QWORD *)v6;
  }
  else
  {
    while ( v12 != *(__int64 ***)result )
    {
      result += 8;
      if ( a3 == (__int64 ***)result )
        goto LABEL_10;
    }
    if ( (_BYTE)a5 )
      goto LABEL_7;
  }
LABEL_12:
  v16 = a6;
  v58 = a6;
  v17 = sub_103BF70(a1, *a2, v7, a6);
  v12 = (__int64 **)*a2;
  v18 = v17;
  sub_103BCD0(a1, *a2, v17, v16);
  v21 = a2[3];
  v59 = a2;
  v61 = v18;
  v60 = v21;
  v22 = (unsigned int)v63;
  v23 = (unsigned int)v63 + 1LL;
  if ( v23 > HIDWORD(v63) )
  {
    v55 = v62;
    if ( v62 > &v59 )
    {
      v12 = (__int64 **)v64;
      sub_C8D5F0((__int64)&v62, v64, v23, 0x18u, v19, v20);
      v24 = v62;
      v22 = (unsigned int)v63;
      v25 = (const __m128i *)&v59;
    }
    else
    {
      v12 = (__int64 **)v64;
      if ( &v59 < &v62[3 * (unsigned int)v63] )
      {
        sub_C8D5F0((__int64)&v62, v64, v23, 0x18u, v19, v20);
        v24 = v62;
        v22 = (unsigned int)v63;
        v25 = (const __m128i *)((char *)v62 + (char *)&v59 - (char *)v55);
      }
      else
      {
        sub_C8D5F0((__int64)&v62, v64, v23, 0x18u, v19, v20);
        v24 = v62;
        v22 = (unsigned int)v63;
        v25 = (const __m128i *)&v59;
      }
    }
  }
  else
  {
    v24 = v62;
    v25 = (const __m128i *)&v59;
  }
  v26 = (__m128i *)&v24[3 * v22];
  *v26 = _mm_loadu_si128(v25);
  v14 = v62;
  v26[1].m128i_i64[0] = v25[1].m128i_i64[0];
  v11 = (_DWORD)v63 == -1;
  result = (unsigned int)(v63 + 1);
  LODWORD(v63) = v63 + 1;
  if ( !v11 )
  {
    while ( 1 )
    {
      v27 = (__int64)&v14[3 * (unsigned int)result - 3];
      v28 = *(__int64 ***)(v27 + 8);
      v29 = *(unsigned int *)(*(_QWORD *)v27 + 32LL);
      v12 = (__int64 **)(*(_QWORD *)(*(_QWORD *)v27 + 24LL) + 8 * v29);
      if ( v28 != v12 )
        break;
      result = (unsigned int)(result - 1);
      LODWORD(v63) = result;
LABEL_32:
      if ( !(_DWORD)result )
        goto LABEL_33;
    }
    v30 = *v28;
    v31 = (__int64 *)(v28 + 1);
    v32 = *(_QWORD *)(v27 + 16);
    *(_QWORD *)(v27 + 8) = v31;
    v33 = *v30;
    if ( *(_BYTE *)(v6 + 28) )
    {
      v34 = *(__int64 **)(v6 + 8);
      v27 = *(unsigned int *)(v6 + 20);
      v31 = &v34[v27];
      if ( v34 != v31 )
      {
        while ( v33 != *v34 )
        {
          if ( v31 == ++v34 )
            goto LABEL_38;
        }
        if ( !v56 )
          goto LABEL_37;
        goto LABEL_22;
      }
LABEL_38:
      if ( (unsigned int)v27 < *(_DWORD *)(v6 + 16) )
      {
        *(_DWORD *)(v6 + 20) = v27 + 1;
        *v31 = v33;
        ++*(_QWORD *)v6;
        goto LABEL_37;
      }
    }
    sub_C8CC70(v6, *v30, (__int64)v31, v27, v29, v20);
    if ( (v56 & (v51 ^ 1)) == 0 )
    {
LABEL_37:
      v32 = sub_103BF70(a1, v33, v32, v58);
LABEL_29:
      v12 = (__int64 **)v33;
      sub_103BCD0(a1, v33, v32, v58);
      v45 = v30[3];
      v59 = v30;
      v61 = v32;
      v60 = v45;
      v46 = (unsigned int)v63;
      v47 = (unsigned int)v63 + 1LL;
      if ( v47 > HIDWORD(v63) )
      {
        v52 = v62;
        v12 = (__int64 **)v64;
        if ( v62 > &v59 || &v59 >= &v62[3 * (unsigned int)v63] )
        {
          sub_C8D5F0((__int64)&v62, v64, v47, 0x18u, v44, v20);
          v48 = v62;
          v46 = (unsigned int)v63;
          v49 = (const __m128i *)&v59;
        }
        else
        {
          sub_C8D5F0((__int64)&v62, v64, v47, 0x18u, v44, v20);
          v48 = v62;
          v46 = (unsigned int)v63;
          v49 = (const __m128i *)((char *)v62 + (char *)&v59 - (char *)v52);
        }
      }
      else
      {
        v48 = v62;
        v49 = (const __m128i *)&v59;
      }
      v50 = (__m128i *)&v48[3 * v46];
      *v50 = _mm_loadu_si128(v49);
      v14 = v62;
      v50[1].m128i_i64[0] = v49[1].m128i_i64[0];
      result = (unsigned int)(v63 + 1);
      LODWORD(v63) = v63 + 1;
      goto LABEL_32;
    }
LABEL_22:
    v35 = *(unsigned int *)(a1 + 120);
    v36 = *(_QWORD *)(a1 + 104);
    if ( (_DWORD)v35 )
    {
      v37 = (v35 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v38 = (__int64 *)(v36 + 16LL * v37);
      v39 = *v38;
      if ( v33 == *v38 )
      {
LABEL_24:
        if ( v38 != (__int64 *)(v36 + 16 * v35) )
        {
          v40 = (__int64 *)v38[1];
          if ( v40 )
          {
            v41 = *v40;
            v42 = 0;
            v43 = v41 & 0xFFFFFFFFFFFFFFF8LL;
            if ( v43 )
              v42 = v43 - 48;
            v32 = v42;
          }
        }
      }
      else
      {
        v53 = 1;
        while ( v39 != -4096 )
        {
          v54 = v53 + 1;
          v37 = (v35 - 1) & (v53 + v37);
          v38 = (__int64 *)(v36 + 16LL * v37);
          v39 = *v38;
          if ( v33 == *v38 )
            goto LABEL_24;
          v53 = v54;
        }
      }
    }
    goto LABEL_29;
  }
LABEL_33:
  if ( v14 != (__int64 **)v64 )
    return _libc_free(v14, v12);
  return result;
}
