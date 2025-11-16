// Function: sub_26028B0
// Address: 0x26028b0
//
void __fastcall sub_26028B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9)
{
  int v11; // esi
  unsigned __int64 v12; // r14
  __int64 *v13; // rax
  __int64 *v14; // r14
  __int64 *v15; // rbx
  __int64 *i; // rsi
  __int64 *v17; // rbx
  unsigned __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 *v21; // r15
  char *v22; // rax
  __int64 *v23; // rdi
  int v24; // r14d
  char *v25; // rcx
  char v26; // al
  char **v27; // rcx
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rsi
  unsigned int v31; // esi
  __int64 v32; // r8
  unsigned int v33; // edx
  _QWORD *v34; // rax
  __int64 v35; // rcx
  __m128i v36; // xmm1
  int v37; // edx
  __int64 v38; // r12
  int v39; // r9d
  _QWORD *v40; // rdi
  int v41; // eax
  int v42; // edx
  __int64 *v43; // [rsp+0h] [rbp-110h]
  __int64 *v44; // [rsp+8h] [rbp-108h]
  __int64 v45; // [rsp+10h] [rbp-100h]
  __int64 *v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+20h] [rbp-F0h]
  __int64 v49; // [rsp+28h] [rbp-E8h]
  __int64 v50; // [rsp+28h] [rbp-E8h]
  unsigned __int64 v51; // [rsp+30h] [rbp-E0h] BYREF
  __int64 *v52; // [rsp+38h] [rbp-D8h]
  __int64 *v53; // [rsp+40h] [rbp-D0h]
  _OWORD v54[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v55; // [rsp+70h] [rbp-A0h]
  char *v56; // [rsp+80h] [rbp-90h] BYREF
  __int64 v57; // [rsp+88h] [rbp-88h]
  char *v58; // [rsp+90h] [rbp-80h]
  __int16 v59; // [rsp+A0h] [rbp-70h]
  __int64 v60; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v61; // [rsp+B8h] [rbp-58h]
  int v62; // [rsp+C0h] [rbp-50h]
  __int16 v63; // [rsp+D0h] [rbp-40h]

  v11 = *(_DWORD *)(a1 + 16);
  v51 = 0;
  v52 = 0;
  v53 = 0;
  if ( !v11 )
    goto LABEL_2;
  v13 = *(__int64 **)(a1 + 8);
  v14 = &v13[2 * *(unsigned int *)(a1 + 24)];
  if ( v13 == v14 )
    goto LABEL_2;
  while ( 1 )
  {
    v15 = v13;
    if ( *v13 != -4096 && *v13 != -8192 )
      break;
    v13 += 2;
    if ( v14 == v13 )
      goto LABEL_2;
  }
  if ( v14 == v13 )
  {
LABEL_2:
    v43 = 0;
    v44 = 0;
  }
  else
  {
    i = 0;
LABEL_40:
    sub_9281F0((__int64)&v51, i, v15);
    for ( i = v52; ; v52 = i )
    {
      v15 += 2;
      if ( v15 == v14 )
        break;
      while ( *v15 == -8192 || *v15 == -4096 )
      {
        v15 += 2;
        if ( v14 == v15 )
          goto LABEL_18;
      }
      if ( v14 == v15 )
        break;
      if ( v53 == i )
        goto LABEL_40;
      if ( i )
      {
        *i = *v15;
        i = v52;
      }
      ++i;
    }
LABEL_18:
    v17 = (__int64 *)v51;
    v18 = (unsigned __int64)i;
    v19 = (__int64)i - v51;
    if ( (__int64 *)((char *)i - v51) == (__int64 *)8 )
      goto LABEL_24;
    v43 = (__int64 *)v51;
    v44 = i;
    if ( v19 > 0 )
    {
      v20 = v19 >> 3;
      v45 = a3;
      v21 = (__int64 *)v51;
      do
      {
        v22 = (char *)sub_2207800(8 * v20);
        v12 = (unsigned __int64)v22;
        if ( v22 )
        {
          v23 = v21;
          a3 = v45;
          sub_2601DA0(v23, i, v22, v20);
          goto LABEL_23;
        }
        v20 >>= 1;
      }
      while ( v20 );
      a3 = v45;
    }
  }
  v12 = 0;
  sub_25F9220(v43, v44);
LABEL_23:
  j_j___libc_free_0(v12);
  v17 = (__int64 *)v51;
  v18 = (unsigned __int64)v52;
LABEL_24:
  if ( v17 == (__int64 *)v18 )
    goto LABEL_42;
  v46 = (__int64 *)v18;
  v24 = 0;
  do
  {
    v36 = _mm_loadu_si128((const __m128i *)&a8);
    v37 = v24++;
    v26 = a9;
    v38 = *v17;
    v54[0] = _mm_loadu_si128((const __m128i *)&a7);
    v55 = a9;
    v54[1] = v36;
    if ( (_BYTE)a9 )
    {
      if ( (_BYTE)a9 == 1 )
      {
        v27 = (char **)"_";
        v56 = "_";
        v59 = 259;
        v28 = 3;
        v48 = v57;
      }
      else
      {
        if ( BYTE1(v55) == 1 )
        {
          v47 = *((_QWORD *)&v54[0] + 1);
          v25 = *(char **)&v54[0];
        }
        else
        {
          v25 = (char *)v54;
          v26 = 2;
        }
        v56 = v25;
        HIBYTE(v59) = 3;
        v57 = v47;
        v58 = "_";
        v27 = &v56;
        LOBYTE(v59) = v26;
        v28 = 2;
      }
      v60 = (__int64)v27;
      v62 = v37;
      v61 = v48;
      LOBYTE(v63) = v28;
      HIBYTE(v63) = 9;
    }
    else
    {
      v59 = 256;
      v63 = 256;
    }
    v49 = sub_B2BE50(a3);
    v29 = sub_22077B0(0x50u);
    if ( v29 )
    {
      v30 = v49;
      v50 = v29;
      sub_AA4D50(v29, v30, (__int64)&v60, a3, 0);
      v29 = v50;
    }
    v31 = *(_DWORD *)(a2 + 24);
    v60 = v38;
    v61 = v29;
    if ( !v31 )
    {
      ++*(_QWORD *)a2;
      v56 = 0;
LABEL_56:
      v31 *= 2;
LABEL_57:
      sub_26026D0(a2, v31);
      sub_25FD8E0(a2, &v60, &v56);
      v38 = v60;
      v40 = v56;
      v42 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_52;
    }
    v32 = *(_QWORD *)(a2 + 8);
    v33 = (v31 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
    v34 = (_QWORD *)(v32 + 16LL * v33);
    v35 = *v34;
    if ( v38 == *v34 )
      goto LABEL_35;
    v39 = 1;
    v40 = 0;
    while ( v35 != -4096 )
    {
      if ( v35 != -8192 || v40 )
        v34 = v40;
      v33 = (v31 - 1) & (v39 + v33);
      v35 = *(_QWORD *)(v32 + 16LL * v33);
      if ( v38 == v35 )
        goto LABEL_35;
      ++v39;
      v40 = v34;
      v34 = (_QWORD *)(v32 + 16LL * v33);
    }
    if ( !v40 )
      v40 = v34;
    v41 = *(_DWORD *)(a2 + 16);
    ++*(_QWORD *)a2;
    v42 = v41 + 1;
    v56 = (char *)v40;
    if ( 4 * (v41 + 1) >= 3 * v31 )
      goto LABEL_56;
    if ( v31 - *(_DWORD *)(a2 + 20) - v42 <= v31 >> 3 )
      goto LABEL_57;
LABEL_52:
    *(_DWORD *)(a2 + 16) = v42;
    if ( *v40 != -4096 )
      --*(_DWORD *)(a2 + 20);
    *v40 = v38;
    v40[1] = v61;
LABEL_35:
    ++v17;
  }
  while ( v46 != v17 );
  v18 = v51;
LABEL_42:
  if ( v18 )
    j_j___libc_free_0(v18);
}
