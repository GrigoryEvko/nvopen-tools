// Function: sub_3911230
// Address: 0x3911230
//
void *__fastcall sub_3911230(__int64 a1, __int64 **a2, size_t a3)
{
  size_t v3; // r14
  __int64 v4; // rax
  __int64 *v5; // rcx
  __int64 *v6; // r13
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // r15
  __int64 v10; // rcx
  unsigned int v11; // eax
  __int64 v12; // rax
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rdx
  __int64 v16; // r12
  _BYTE *v17; // rdi
  __int64 v18; // rdx
  size_t v19; // r13
  __m128i *v20; // r14
  unsigned int v21; // eax
  __int64 v22; // r12
  unsigned int v23; // ebx
  __m128i *v24; // rbx
  __int64 v25; // r12
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // r12
  int v28; // r9d
  int v29; // r8d
  int v30; // eax
  __int64 v31; // rax
  __m128i *v32; // rax
  int v33; // r8d
  int v34; // r9d
  int v35; // eax
  __int64 v36; // rax
  __m128i *v37; // rax
  unsigned int v38; // eax
  __int64 v39; // r14
  int v40; // r12d
  int *v41; // rax
  int v42; // ebx
  __int64 v44; // [rsp+10h] [rbp-150h]
  __int64 v45; // [rsp+18h] [rbp-148h]
  __int64 v46; // [rsp+20h] [rbp-140h]
  __int64 v47; // [rsp+28h] [rbp-138h]
  __int64 v48; // [rsp+30h] [rbp-130h]
  __int64 v49; // [rsp+38h] [rbp-128h]
  __int64 v50; // [rsp+50h] [rbp-110h]
  __int16 v51; // [rsp+58h] [rbp-108h]
  char *src; // [rsp+60h] [rbp-100h]
  __int16 v53; // [rsp+6Eh] [rbp-F2h]
  int v54; // [rsp+70h] [rbp-F0h]
  __m128i *v55; // [rsp+70h] [rbp-F0h]
  size_t n; // [rsp+78h] [rbp-E8h]
  size_t na; // [rsp+78h] [rbp-E8h]
  __int64 v58; // [rsp+80h] [rbp-E0h]
  unsigned int v59; // [rsp+80h] [rbp-E0h]
  __int64 *v60; // [rsp+88h] [rbp-D8h]
  unsigned int v61; // [rsp+88h] [rbp-D8h]
  int v62; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v63; // [rsp+90h] [rbp-D0h] BYREF
  int v64; // [rsp+98h] [rbp-C8h]
  int v65; // [rsp+9Ch] [rbp-C4h]
  __int64 v66; // [rsp+A0h] [rbp-C0h]
  _QWORD v67[3]; // [rsp+B0h] [rbp-B0h] BYREF
  int v68; // [rsp+C8h] [rbp-98h]
  _QWORD v69[2]; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v70; // [rsp+E0h] [rbp-80h]
  void *dest; // [rsp+E8h] [rbp-78h]
  int v72; // [rsp+F0h] [rbp-70h]
  size_t v73; // [rsp+F8h] [rbp-68h]
  _BYTE *v74; // [rsp+100h] [rbp-60h] BYREF
  __int64 v75; // [rsp+108h] [rbp-58h]
  _BYTE v76[80]; // [rsp+110h] [rbp-50h] BYREF

  v3 = a3;
  v4 = **a2;
  *(_DWORD *)(a3 + 72) = 0;
  *(_DWORD *)(a3 + 120) = 0;
  v44 = v4;
  v72 = 1;
  dest = 0;
  v69[0] = &unk_49EFC48;
  v73 = a3 + 64;
  v70 = 0;
  v69[1] = 0;
  sub_16E7A40((__int64)v69, 0, 0, 0);
  v5 = *(__int64 **)(v3 + 224);
  v74 = v76;
  v75 = 0x400000000LL;
  v48 = *(unsigned int *)(v3 + 232);
  v60 = &v5[2 * v48];
  if ( v60 == v5 )
  {
    if ( !*(_DWORD *)(v3 + 232) )
      goto LABEL_39;
    v17 = v76;
    goto LABEL_11;
  }
  v6 = v5 + 2;
  v7 = 0;
  n = v3;
  v8 = v5[1];
  v9 = *v5;
  while ( 1 )
  {
    v12 = sub_390F830(a2, v9, v8);
    v15 = (unsigned int)v75;
    v16 = (v12 << 32) | v7;
    if ( (unsigned int)v75 >= HIDWORD(v75) )
    {
      sub_16CD150((__int64)&v74, v76, 0, 8, v13, v14);
      v15 = (unsigned int)v75;
    }
    *(_QWORD *)&v74[8 * v15] = v16;
    LODWORD(v75) = v75 + 1;
    if ( v60 == v6 )
      break;
    v9 = *v6;
    v10 = v6[1];
    v7 = 0;
    if ( v8 )
    {
      v58 = v6[1];
      v11 = sub_390F830(a2, v8, *v6);
      v10 = v58;
      v7 = v11;
    }
    v6 += 2;
    v8 = v10;
  }
  v3 = n;
  v17 = v74;
  v48 = *(unsigned int *)(n + 232);
  if ( *(_DWORD *)(n + 232) )
  {
    v5 = *(__int64 **)(n + 224);
LABEL_11:
    v18 = 0;
    v45 = v3 + 112;
    v19 = v3;
    v20 = (__m128i *)&v63;
    while ( 1 )
    {
      v49 = v18 + 1;
      v50 = v5[2 * v18];
      v47 = 8 * v18;
      v21 = *(_DWORD *)&v17[8 * v18 + 4];
      if ( v18 + 1 == v48 )
      {
        v22 = v48;
        v23 = *(_DWORD *)&v17[8 * v18 + 4];
      }
      else
      {
        v22 = v18 + 1;
        while ( 1 )
        {
          v23 = v21;
          v21 += *(_DWORD *)&v17[8 * v22] + *(_DWORD *)&v17[8 * v22 + 4];
          if ( v21 > 0xF000 )
            break;
          if ( ++v22 == v48 )
          {
            v23 = v21;
            break;
          }
        }
      }
      v61 = v23;
      v24 = v20;
      v46 = v22;
      v59 = 0;
      v51 = 4 * (v22 + ~(_WORD)v18);
      do
      {
        if ( v61 > 0xEFFF )
        {
          v54 = 61440;
          v61 -= 61440;
          v53 = -4096;
        }
        else
        {
          v38 = v61;
          v61 = 0;
          v53 = v38;
          v54 = v38;
        }
        v25 = sub_38CF310(v50, 0, v44, 0);
        v26 = sub_38CB470(v59, v44);
        v27 = sub_38CB1F0(0, v25, v26, v44, 0);
        memset(v67, 0, sizeof(v67));
        v68 = 0;
        sub_38CF2C0(v27, (__int64)v67, a2, 0);
        src = *(char **)(v19 + 272);
        na = *(unsigned int *)(v19 + 280);
        LOWORD(v63) = v51 + na + 8;
        sub_16E7EE0((__int64)v69, v20->m128i_i8, 2u);
        v29 = na;
        if ( na > v70 - (__int64)dest )
        {
          sub_16E7EE0((__int64)v69, src, na);
        }
        else if ( na )
        {
          memcpy(dest, src, na);
          v29 = na;
          dest = (char *)dest + na;
        }
        v30 = *(_DWORD *)(v19 + 72);
        v63 = v27;
        v65 = 18;
        v64 = v30;
        v31 = *(unsigned int *)(v19 + 120);
        v66 = 0;
        if ( (unsigned int)v31 >= *(_DWORD *)(v19 + 124) )
        {
          sub_16CD150(v45, (const void *)(v19 + 128), 0, 24, v29, v28);
          v31 = *(unsigned int *)(v19 + 120);
        }
        v32 = (__m128i *)(*(_QWORD *)(v19 + 112) + 24 * v31);
        *v32 = _mm_loadu_si128(v20);
        v32[1].m128i_i64[0] = v20[1].m128i_i64[0];
        ++*(_DWORD *)(v19 + 120);
        LODWORD(v63) = 0;
        sub_16E7EE0((__int64)v69, v20->m128i_i8, 4u);
        v35 = *(_DWORD *)(v19 + 72);
        v63 = v27;
        v65 = 17;
        v64 = v35;
        v36 = *(unsigned int *)(v19 + 120);
        v66 = 0;
        if ( (unsigned int)v36 >= *(_DWORD *)(v19 + 124) )
        {
          sub_16CD150(v45, (const void *)(v19 + 128), 0, 24, v33, v34);
          v36 = *(unsigned int *)(v19 + 120);
        }
        v37 = (__m128i *)(*(_QWORD *)(v19 + 112) + 24 * v36);
        *v37 = _mm_loadu_si128(v20);
        v37[1].m128i_i64[0] = v20[1].m128i_i64[0];
        ++*(_DWORD *)(v19 + 120);
        LOWORD(v63) = 0;
        sub_16E7EE0((__int64)v69, v20->m128i_i8, 2u);
        LOWORD(v63) = v53;
        sub_16E7EE0((__int64)v69, v20->m128i_i8, 2u);
        v59 += v54;
      }
      while ( v61 );
      v17 = v74;
      if ( v46 != v49 )
      {
        v39 = v49;
        v40 = *(_DWORD *)&v74[v47 + 4];
        v55 = v24;
        while ( 1 )
        {
          v41 = (int *)&v17[8 * v39];
          v42 = *v41;
          LODWORD(v41) = v41[1];
          LOWORD(v67[0]) = v40;
          ++v39;
          v62 = (int)v41;
          sub_16E7EE0((__int64)v69, (char *)v67, 2u);
          LOWORD(v67[0]) = v42;
          sub_16E7EE0((__int64)v69, (char *)v67, 2u);
          v40 += v62 + v42;
          if ( v46 == v39 )
            break;
          v17 = v74;
        }
        v49 = v39;
        v20 = v55;
        v17 = v74;
      }
      v18 = v49;
      if ( v49 == v48 )
        break;
      v5 = *(__int64 **)(v19 + 224);
    }
  }
  if ( v17 != v76 )
    _libc_free((unsigned __int64)v17);
LABEL_39:
  v69[0] = &unk_49EFD28;
  return sub_16E7960((__int64)v69);
}
