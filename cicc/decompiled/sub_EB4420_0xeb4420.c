// Function: sub_EB4420
// Address: 0xeb4420
//
__int64 __fastcall sub_EB4420(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rax
  unsigned int v4; // edx
  _DWORD *v5; // rdx
  int v6; // eax
  int v7; // eax
  _DWORD *i; // rbx
  unsigned __int64 v9; // rcx
  int v10; // eax
  unsigned __int64 v11; // r12
  __m128i v12; // xmm0
  bool v13; // cc
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  _DWORD *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  int v23; // eax
  unsigned __int64 v24; // r12
  __m128i v25; // xmm1
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rax
  _DWORD *v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 *v34; // rdi
  __int64 v35; // r12
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // edx
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // rdx
  __int64 v49; // rax
  const char *v50; // r13
  const char *v51; // rbx
  __int64 v52; // r14
  __int64 v53; // r12
  __int64 v54; // rdi
  __int64 v55; // r12
  __int64 v56; // rax
  unsigned __int64 v57; // rax
  __int64 *v58; // rdi
  __int64 v59; // rax
  int v60; // [rsp+Ch] [rbp-E4h]
  __int64 v61[2]; // [rsp+10h] [rbp-E0h] BYREF
  __int64 v62; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v63; // [rsp+28h] [rbp-C8h]
  int v64; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+38h] [rbp-B8h]
  __int64 v66; // [rsp+48h] [rbp-A8h] BYREF
  unsigned int v67; // [rsp+50h] [rbp-A0h]
  int v68; // [rsp+60h] [rbp-90h] BYREF
  __m128i v69; // [rsp+68h] [rbp-88h]
  __int64 v70; // [rsp+78h] [rbp-78h] BYREF
  unsigned int v71; // [rsp+80h] [rbp-70h]
  const char *v72; // [rsp+90h] [rbp-60h] BYREF
  const char *v73; // [rsp+98h] [rbp-58h]
  __int64 v74; // [rsp+A0h] [rbp-50h]
  __int64 v75; // [rsp+A8h] [rbp-48h]
  unsigned int v76; // [rsp+B0h] [rbp-40h]

  v64 = 0;
  v65 = 0u;
  v67 = 1;
  v66 = 0;
  v3 = sub_ECD7B0(a1);
  v68 = *(_DWORD *)v3;
  v4 = *(_DWORD *)(v3 + 32);
  v69 = _mm_loadu_si128((const __m128i *)(v3 + 8));
  v71 = v4;
  if ( v4 > 0x40 )
    sub_C43780((__int64)&v70, (const void **)(v3 + 24));
  else
    v70 = *(_QWORD *)(v3 + 24);
  v60 = 0;
  v5 = *(_DWORD **)(a1 + 48);
  v6 = *v5;
  if ( !*v5 )
  {
LABEL_35:
    v34 = *(__int64 **)(a1 + 248);
    v72 = "no matching '.endr' in definition";
    LOWORD(v76) = 259;
    *(_BYTE *)(a1 + 32) = 1;
    v62 = 0;
    v63 = 0;
    sub_C91CB0(v34, a2, 0, (__int64)&v72, (__int64)&v62, 1, 0, 0, 1u);
LABEL_36:
    v35 = 0;
    sub_EA2AE0((_QWORD *)a1);
    goto LABEL_37;
  }
  while ( 1 )
  {
    if ( v6 != 2 )
      goto LABEL_5;
    v37 = sub_ECD7B0(a1);
    v38 = v37;
    if ( *(_DWORD *)v37 == 2 )
    {
      v41 = *(_QWORD *)(v37 + 8);
      v42 = *(_QWORD *)(v37 + 16);
    }
    else
    {
      v39 = *(_QWORD *)(v37 + 16);
      if ( !v39 )
        goto LABEL_20;
      v40 = v39 - 1;
      if ( !v40 )
        v40 = 1;
      v41 = *(_QWORD *)(v38 + 8) + 1LL;
      v42 = v40 - 1;
    }
    if ( v42 == 4 )
    {
      if ( *(_DWORD *)v41 != 1885696558 && *(_DWORD *)v41 != 1886546222 )
        goto LABEL_20;
      goto LABEL_58;
    }
    if ( v42 != 5 )
      goto LABEL_20;
    if ( *(_DWORD *)v41 != 1885696558 )
    {
      if ( *(_DWORD *)v41 != 1886546222 )
        goto LABEL_53;
      goto LABEL_64;
    }
    if ( *(_BYTE *)(v41 + 4) != 116 )
      break;
LABEL_58:
    ++v60;
    v5 = *(_DWORD **)(a1 + 48);
LABEL_5:
    v7 = *v5;
    for ( i = v5; *v5 != 9; i = v5 )
    {
      if ( !v7 )
        goto LABEL_35;
      v9 = *(unsigned int *)(a1 + 56);
      *(_BYTE *)(a1 + 155) = 0;
      v10 = v9;
      v9 *= 40LL;
      v11 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v9 - 40) >> 3);
      if ( v9 > 0x28 )
      {
        do
        {
          v12 = _mm_loadu_si128((const __m128i *)i + 3);
          v13 = i[8] <= 0x40u;
          *i = i[10];
          *(__m128i *)(i + 2) = v12;
          if ( !v13 )
          {
            v14 = *((_QWORD *)i + 3);
            if ( v14 )
              j_j___libc_free_0_0(v14);
          }
          v15 = *((_QWORD *)i + 8);
          i += 10;
          *((_QWORD *)i - 2) = v15;
          LODWORD(v15) = i[8];
          i[8] = 0;
          *(i - 2) = v15;
          --v11;
        }
        while ( v11 );
        v10 = *(_DWORD *)(a1 + 56);
        v5 = *(_DWORD **)(a1 + 48);
      }
      v16 = (unsigned int)(v10 - 1);
      *(_DWORD *)(a1 + 56) = v16;
      v17 = &v5[10 * v16];
      if ( v17[8] > 0x40u )
      {
        v18 = *((_QWORD *)v17 + 3);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      if ( !*(_DWORD *)(a1 + 56) )
      {
        sub_1097F60(&v72, a1 + 40);
        sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v72, v19, v20, v21);
        if ( v76 > 0x40 )
        {
          if ( v75 )
            j_j___libc_free_0_0(v75);
        }
      }
LABEL_20:
      v5 = *(_DWORD **)(a1 + 48);
      v7 = *v5;
    }
    v22 = *(unsigned int *)(a1 + 56);
    *(_BYTE *)(a1 + 155) = 1;
    v23 = v22;
    v22 *= 40LL;
    v24 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v22 - 40) >> 3);
    if ( v22 > 0x28 )
    {
      do
      {
        v25 = _mm_loadu_si128((const __m128i *)i + 3);
        v13 = i[8] <= 0x40u;
        *i = i[10];
        *(__m128i *)(i + 2) = v25;
        if ( !v13 )
        {
          v26 = *((_QWORD *)i + 3);
          if ( v26 )
            j_j___libc_free_0_0(v26);
        }
        v27 = *((_QWORD *)i + 8);
        i += 10;
        *((_QWORD *)i - 2) = v27;
        LODWORD(v27) = i[8];
        i[8] = 0;
        *(i - 2) = v27;
        --v24;
      }
      while ( v24 );
      v23 = *(_DWORD *)(a1 + 56);
      i = *(_DWORD **)(a1 + 48);
    }
    v28 = (unsigned int)(v23 - 1);
    *(_DWORD *)(a1 + 56) = v28;
    v29 = &i[10 * v28];
    if ( v29[8] > 0x40u )
    {
      v30 = *((_QWORD *)v29 + 3);
      if ( v30 )
        j_j___libc_free_0_0(v30);
    }
    if ( !*(_DWORD *)(a1 + 56) )
    {
      sub_1097F60(&v72, a1 + 40);
      sub_EAA0A0(a1 + 48, *(_QWORD *)(a1 + 48), (unsigned __int64)&v72, v31, v32, v33);
      if ( v76 > 0x40 )
      {
        if ( v75 )
          j_j___libc_free_0_0(v75);
      }
    }
    v5 = *(_DWORD **)(a1 + 48);
    v6 = *v5;
    if ( !*v5 )
      goto LABEL_35;
  }
  if ( *(_DWORD *)v41 != 1886546222 )
    goto LABEL_53;
LABEL_64:
  if ( *(_BYTE *)(v41 + 4) == 99 )
    goto LABEL_58;
LABEL_53:
  if ( *(_DWORD *)v41 != 1684956462 || *(_BYTE *)(v41 + 4) != 114 )
    goto LABEL_20;
  if ( v60 )
  {
    --v60;
    v5 = *(_DWORD **)(a1 + 48);
    goto LABEL_5;
  }
  v43 = sub_ECD7B0(a1);
  v64 = *(_DWORD *)v43;
  v65 = _mm_loadu_si128((const __m128i *)(v43 + 8));
  if ( v67 <= 0x40 && (v44 = *(_DWORD *)(v43 + 32), v44 <= 0x40) )
  {
    v59 = *(_QWORD *)(v43 + 24);
    v67 = v44;
    v66 = v59;
  }
  else
  {
    sub_C43990((__int64)&v66, v43 + 24);
  }
  sub_EABFE0(a1);
  if ( **(_DWORD **)(a1 + 48) != 9 )
  {
    v72 = "expected newline";
    LOWORD(v76) = 259;
    v56 = sub_ECD7B0(a1);
    v57 = sub_ECD6A0(v56);
    *(_BYTE *)(a1 + 32) = 1;
    v58 = *(__int64 **)(a1 + 248);
    v62 = 0;
    v63 = 0;
    sub_C91CB0(v58, v57, 0, (__int64)&v72, (__int64)&v62, 1, 0, 0, 1u);
    goto LABEL_36;
  }
  v45 = sub_ECD6A0(&v68);
  v46 = sub_ECD6A0(&v64);
  v47 = *(_QWORD *)(a1 + 456);
  v61[0] = v45;
  v72 = 0;
  v61[1] = v46 - v45;
  v48 = v46 - v45;
  v49 = *(_QWORD *)(a1 + 440);
  v73 = 0;
  v74 = 0;
  v62 = 0;
  v63 = 0;
  if ( v49 == v47 - 88 )
  {
    sub_EA93B0((__int64 *)(a1 + 392), &v62, v61, &v72);
    v50 = v73;
    v51 = v72;
  }
  else
  {
    if ( v49 )
    {
      *(_QWORD *)v49 = 0;
      *(_QWORD *)(v49 + 8) = 0;
      *(_QWORD *)(v49 + 16) = v45;
      *(_QWORD *)(v49 + 24) = v48;
      *(_QWORD *)(v49 + 32) = 0;
      *(_QWORD *)(v49 + 40) = 0;
      *(_QWORD *)(v49 + 48) = 0;
      *(_QWORD *)(v49 + 56) = 0;
      *(_QWORD *)(v49 + 64) = 0;
      *(_QWORD *)(v49 + 72) = 0;
      *(_BYTE *)(v49 + 80) = 0;
      *(_DWORD *)(v49 + 84) = 0;
      v49 = *(_QWORD *)(a1 + 440);
      v50 = v73;
      v51 = v72;
    }
    else
    {
      v51 = 0;
      v50 = 0;
    }
    *(_QWORD *)(a1 + 440) = v49 + 88;
  }
  for ( ; v51 != v50; v51 += 48 )
  {
    v52 = *((_QWORD *)v51 + 3);
    v53 = *((_QWORD *)v51 + 2);
    if ( v52 != v53 )
    {
      do
      {
        if ( *(_DWORD *)(v53 + 32) > 0x40u )
        {
          v54 = *(_QWORD *)(v53 + 24);
          if ( v54 )
            j_j___libc_free_0_0(v54);
        }
        v53 += 40;
      }
      while ( v52 != v53 );
      v53 = *((_QWORD *)v51 + 2);
    }
    if ( v53 )
      j_j___libc_free_0(v53, *((_QWORD *)v51 + 4) - v53);
  }
  if ( v72 )
    j_j___libc_free_0(v72, v74 - (_QWORD)v72);
  v55 = *(_QWORD *)(a1 + 440);
  if ( v55 == *(_QWORD *)(a1 + 448) )
    v55 = *(_QWORD *)(*(_QWORD *)(a1 + 464) - 8LL) + 440LL;
  v35 = v55 - 88;
LABEL_37:
  if ( v71 > 0x40 && v70 )
    j_j___libc_free_0_0(v70);
  if ( v67 > 0x40 && v66 )
    j_j___libc_free_0_0(v66);
  return v35;
}
