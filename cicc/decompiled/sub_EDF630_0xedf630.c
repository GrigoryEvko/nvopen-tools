// Function: sub_EDF630
// Address: 0xedf630
//
__int64 __fastcall sub_EDF630(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rbx
  __int64 v8; // rax
  _WORD *v9; // rdx
  _QWORD *v10; // rax
  int v11; // r10d
  int v13; // esi
  __int64 v14; // rdi
  _QWORD *v15; // rdx
  __int64 *v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 *v34; // r12
  __int64 v35; // rdi
  __int64 v36; // rbx
  __int64 v37; // r14
  _QWORD *v38; // r15
  __int64 *v39; // r12
  __int64 v40; // rdi
  __int64 v41; // rbx
  __int64 v42; // r15
  _QWORD *v43; // r14
  __int64 v44; // rdx
  unsigned int v45; // ecx
  __int64 v46; // rdi
  __int64 v47; // rbx
  __int64 v48; // r15
  _QWORD *v49; // r14
  __int64 *v50; // r12
  __int64 v51; // rdi
  __int64 v52; // rbx
  __int64 v53; // r15
  _QWORD *v54; // r14
  __int32 v55; // r8d
  unsigned int v56; // edi
  __int64 *v57; // [rsp+10h] [rbp-1E0h]
  __int64 v58; // [rsp+10h] [rbp-1E0h]
  __int64 *v59; // [rsp+10h] [rbp-1E0h]
  __int64 v60; // [rsp+10h] [rbp-1E0h]
  __int64 v61; // [rsp+18h] [rbp-1D8h] BYREF
  __int64 v62; // [rsp+28h] [rbp-1C8h] BYREF
  __int64 v63; // [rsp+30h] [rbp-1C0h] BYREF
  char v64; // [rsp+38h] [rbp-1B8h]
  __int64 v65; // [rsp+40h] [rbp-1B0h]
  __m128i v66; // [rsp+50h] [rbp-1A0h] BYREF
  __int64 v67; // [rsp+60h] [rbp-190h]
  __int64 (__fastcall *v68)(__int64, __int64, __int64); // [rsp+68h] [rbp-188h]
  __int64 *v69; // [rsp+70h] [rbp-180h]
  __m128i v70; // [rsp+80h] [rbp-170h] BYREF
  __m128i *v71; // [rsp+90h] [rbp-160h]
  __int16 v72; // [rsp+A0h] [rbp-150h]
  __m128i v73; // [rsp+B0h] [rbp-140h] BYREF
  _QWORD v74[2]; // [rsp+C0h] [rbp-130h] BYREF
  char v75; // [rsp+D0h] [rbp-120h]
  char v76; // [rsp+D1h] [rbp-11Fh]
  _QWORD v77[2]; // [rsp+D8h] [rbp-118h] BYREF
  _QWORD v78[2]; // [rsp+E8h] [rbp-108h] BYREF
  _QWORD v79[2]; // [rsp+F8h] [rbp-F8h] BYREF
  _QWORD v80[12]; // [rsp+108h] [rbp-E8h] BYREF
  char *v81[2]; // [rsp+168h] [rbp-88h] BYREF
  __int64 *v82; // [rsp+178h] [rbp-78h] BYREF
  unsigned int v83; // [rsp+180h] [rbp-70h]
  _BYTE v84[104]; // [rsp+188h] [rbp-68h] BYREF

  v7 = (_QWORD *)a2[31];
  v61 = a3;
  if ( !v7 )
  {
    v76 = 1;
    v73.m128i_i64[0] = (__int64)"no memprof data available in profile";
    v75 = 3;
    v29 = sub_22077B0(48);
    v26 = v29;
    if ( v29 )
    {
      *(_DWORD *)(v29 + 8) = 14;
      *(_QWORD *)v29 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v29 + 16), (void **)&v73);
      *(_BYTE *)(a1 + 264) |= 3u;
      *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
      return a1;
    }
    goto LABEL_15;
  }
  v8 = *(_QWORD *)(v7[2] + 8 * (v61 & (*v7 - 1LL)));
  if ( !v8 || (v9 = (_WORD *)(v7[3] + v8), v10 = v9 + 1, v11 = (unsigned __int16)*v9, !*v9) )
  {
LABEL_17:
    LOWORD(v69) = 267;
    v66.m128i_i64[0] = (__int64)&v61;
    v70.m128i_i64[0] = (__int64)"memprof record not found for function hash ";
    v72 = 259;
    sub_9C6370(&v73, &v70, &v66, v61, a5, a6);
    sub_ED7960(&v63, 13, (void **)&v73);
    v27 = v63;
    *(_BYTE *)(a1 + 264) |= 3u;
    *(_QWORD *)a1 = v27 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v13 = 0;
  while ( 1 )
  {
    v14 = v10[1];
    v15 = v10 + 3;
    a6 = v14 + v10[2];
    if ( v61 == *v10 && v61 == v10[3] )
      break;
    ++v13;
    v10 = (_QWORD *)((char *)v15 + a6);
    if ( v11 == v13 )
      goto LABEL_17;
  }
  v16 = v7 + 35;
  sub_C16570(v73.m128i_i64, (__int64)(v7 + 5), (_QWORD *)((char *)v15 + v14), v7[4]);
  sub_ED66E0((__int64)(v7 + 35), (char **)&v73, v17, v18, v19, v20);
  sub_ED6840((__int64)(v7 + 58), v81, v21, v22, v23, v24);
  if ( (__int64 **)v81[0] != &v82 )
    _libc_free(v81[0], v81);
  if ( (_QWORD *)v73.m128i_i64[0] != v74 )
    _libc_free(v73.m128i_i64[0], v81);
  if ( *a2 == 2 )
  {
    v30 = a2[33];
    v31 = a2[32];
    v32 = (__int64)(v7 + 35);
    v64 = 0;
    v65 = v31;
    v67 = v30;
    v66.m128i_i8[8] = 0;
    v68 = sub_ED6100;
    v69 = &v63;
    sub_C17210(&v73, v16, (void (__fastcall *)(__int64 *, __int64, _QWORD))sub_EDE840, (__int64)&v66);
    if ( v66.m128i_i8[8] )
    {
      v71 = &v66;
      v70.m128i_i64[0] = (__int64)"memprof call stack not found for call stack id ";
    }
    else
    {
      if ( !v64 )
      {
        v55 = v73.m128i_i32[2];
        *(_BYTE *)(a1 + 264) = *(_BYTE *)(a1 + 264) & 0xFC | 2;
        *(_QWORD *)a1 = a1 + 16;
        *(_QWORD *)(a1 + 8) = 0x100000000LL;
        if ( v55 )
        {
          v32 = (__int64)&v73;
          sub_EDE860(a1, (__int64)&v73);
        }
        v56 = v83;
        *(_QWORD *)(a1 + 200) = a1 + 216;
        *(_QWORD *)(a1 + 208) = 0x200000000LL;
        if ( !v56 )
          goto LABEL_36;
        v32 = (__int64)&v82;
        sub_EDF010(a1 + 200, (__int64)&v82);
LABEL_25:
        v57 = v82;
        v34 = &v82[3 * v83];
        if ( v82 == v34 )
        {
LABEL_37:
          if ( v57 != (__int64 *)v84 )
            _libc_free(v57, v32);
          v58 = v73.m128i_i64[0];
          v39 = (__int64 *)(v73.m128i_i64[0] + 184LL * v73.m128i_u32[2]);
          if ( (__int64 *)v73.m128i_i64[0] == v39 )
            goto LABEL_51;
          do
          {
            v40 = *(v39 - 23);
            v41 = *(v39 - 22);
            v39 -= 23;
            v42 = v40;
            if ( v41 != v40 )
            {
              do
              {
                v43 = *(_QWORD **)(v42 + 8);
                if ( v43 )
                {
                  if ( (_QWORD *)*v43 != v43 + 2 )
                    j_j___libc_free_0(*v43, v43[2] + 1LL);
                  v32 = 32;
                  j_j___libc_free_0(v43, 32);
                }
                v42 += 32;
              }
              while ( v41 != v42 );
              v40 = *v39;
            }
            if ( v40 )
            {
              v32 = v39[2] - v40;
              j_j___libc_free_0(v40, v32);
            }
          }
          while ( (__int64 *)v58 != v39 );
          goto LABEL_50;
        }
        do
        {
          v35 = *(v34 - 3);
          v36 = *(v34 - 2);
          v34 -= 3;
          v37 = v35;
          if ( v36 != v35 )
          {
            do
            {
              v38 = *(_QWORD **)(v37 + 8);
              if ( v38 )
              {
                if ( (_QWORD *)*v38 != v38 + 2 )
                  j_j___libc_free_0(*v38, v38[2] + 1LL);
                v32 = 32;
                j_j___libc_free_0(v38, 32);
              }
              v37 += 32;
            }
            while ( v36 != v37 );
            v35 = *v34;
          }
          if ( v35 )
          {
            v32 = v34[2] - v35;
            j_j___libc_free_0(v35, v32);
          }
        }
        while ( v57 != v34 );
LABEL_36:
        v57 = v82;
        goto LABEL_37;
      }
      v71 = (__m128i *)&v63;
      v70.m128i_i64[0] = (__int64)"memprof frame not found for frame id ";
    }
    v32 = 15;
    v72 = 2819;
    sub_ED7960(&v62, 15, (void **)&v70);
    v33 = v62;
    *(_BYTE *)(a1 + 264) |= 3u;
    *(_QWORD *)a1 = v33 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_25;
  }
  if ( *a2 != 3 )
  {
    v73.m128i_i64[0] = (__int64)"MemProf version {} not supported; requires version between {} and {}, inclusive";
    v74[0] = v80;
    v73.m128i_i64[1] = 79;
    v75 = 1;
    v77[0] = &unk_49E4CE8;
    v78[0] = &unk_49E4CE8;
    v78[1] = &unk_3F87948;
    v74[1] = 3;
    v77[1] = &unk_3F87940;
    v79[0] = &unk_49E4E88;
    v80[0] = v79;
    v80[1] = v78;
    v79[1] = a2;
    v80[2] = v77;
    v72 = 263;
    v70.m128i_i64[0] = (__int64)&v73;
    v25 = sub_22077B0(48);
    v26 = v25;
    if ( v25 )
    {
      *(_DWORD *)(v25 + 8) = 5;
      *(_QWORD *)v25 = &unk_49E4BC8;
      sub_CA0F50((__int64 *)(v25 + 16), (void **)&v70);
    }
LABEL_15:
    *(_BYTE *)(a1 + 264) |= 3u;
    *(_QWORD *)a1 = v26 & 0xFFFFFFFFFFFFFFFELL;
    return a1;
  }
  v44 = a2[34];
  v70.m128i_i64[0] = a2[35];
  v66.m128i_i64[0] = v44;
  v70.m128i_i64[1] = (__int64)sub_ED6B00;
  v71 = &v66;
  sub_C17210(&v73, v16, (void (__fastcall *)(__int64 *, __int64, _QWORD))sub_EDB710, (__int64)&v70);
  v32 = v73.m128i_u32[2];
  *(_BYTE *)(a1 + 264) = *(_BYTE *)(a1 + 264) & 0xFC | 2;
  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0x100000000LL;
  if ( (_DWORD)v32 )
  {
    v32 = (__int64)&v73;
    sub_EDE860(a1, (__int64)&v73);
  }
  v45 = v83;
  *(_QWORD *)(a1 + 200) = a1 + 216;
  *(_QWORD *)(a1 + 208) = 0x200000000LL;
  if ( v45 )
  {
    v32 = (__int64)&v82;
    sub_EDF010(a1 + 200, (__int64)&v82);
    v59 = v82;
    v50 = &v82[3 * v83];
    if ( v82 == v50 )
      goto LABEL_57;
    do
    {
      v51 = *(v50 - 3);
      v52 = *(v50 - 2);
      v50 -= 3;
      v53 = v51;
      if ( v52 != v51 )
      {
        do
        {
          v54 = *(_QWORD **)(v53 + 8);
          if ( v54 )
          {
            if ( (_QWORD *)*v54 != v54 + 2 )
              j_j___libc_free_0(*v54, v54[2] + 1LL);
            v32 = 32;
            j_j___libc_free_0(v54, 32);
          }
          v53 += 32;
        }
        while ( v52 != v53 );
        v51 = *v50;
      }
      if ( v51 )
      {
        v32 = v50[2] - v51;
        j_j___libc_free_0(v51, v32);
      }
    }
    while ( v59 != v50 );
  }
  v59 = v82;
LABEL_57:
  if ( v59 != (__int64 *)v84 )
    _libc_free(v59, v32);
  v60 = v73.m128i_i64[0];
  v39 = (__int64 *)(v73.m128i_i64[0] + 184LL * v73.m128i_u32[2]);
  if ( (__int64 *)v73.m128i_i64[0] == v39 )
    goto LABEL_51;
  do
  {
    v46 = *(v39 - 23);
    v47 = *(v39 - 22);
    v39 -= 23;
    v48 = v46;
    if ( v47 != v46 )
    {
      do
      {
        v49 = *(_QWORD **)(v48 + 8);
        if ( v49 )
        {
          if ( (_QWORD *)*v49 != v49 + 2 )
            j_j___libc_free_0(*v49, v49[2] + 1LL);
          v32 = 32;
          j_j___libc_free_0(v49, 32);
        }
        v48 += 32;
      }
      while ( v47 != v48 );
      v46 = *v39;
    }
    if ( v46 )
    {
      v32 = v39[2] - v46;
      j_j___libc_free_0(v46, v32);
    }
  }
  while ( (__int64 *)v60 != v39 );
LABEL_50:
  v39 = (__int64 *)v73.m128i_i64[0];
LABEL_51:
  if ( v39 != v74 )
    _libc_free(v39, v32);
  return a1;
}
