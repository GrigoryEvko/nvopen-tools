// Function: sub_37CF130
// Address: 0x37cf130
//
__int64 __fastcall sub_37CF130(__int64 a1, unsigned int a2, const __m128i *a3, __int64 a4, int a5)
{
  __int64 v6; // r15
  unsigned int v9; // esi
  int v10; // eax
  int v11; // r10d
  int *v12; // r13
  __int64 v13; // rcx
  unsigned int v14; // edx
  __int64 v15; // r12
  int v16; // r8d
  __int64 v17; // rdx
  _QWORD *v18; // r13
  int v19; // eax
  __int64 v20; // r12
  __int64 v21; // rax
  const void *v22; // r8
  void *v23; // rdi
  unsigned int v24; // r15d
  size_t v25; // r14
  __int64 v26; // rax
  unsigned __int64 v27; // r10
  unsigned int v28; // esi
  __int64 result; // rax
  __int64 v30; // r8
  unsigned int v31; // edx
  _DWORD *v32; // rdi
  int v33; // ecx
  __int64 v34; // rcx
  const void *v35; // r8
  const __m128i *v36; // r9
  __int64 v37; // r15
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r10
  __int64 v41; // rax
  void *v42; // rdi
  unsigned int v43; // r11d
  signed __int64 v44; // rax
  unsigned __int64 v45; // r14
  int v46; // r13d
  unsigned __int64 v47; // rdi
  int v48; // ecx
  int v49; // ecx
  _DWORD *v50; // rax
  int v51; // r11d
  _DWORD *v52; // r10
  int v53; // ecx
  int v54; // ecx
  signed __int64 v55; // [rsp+8h] [rbp-78h]
  const __m128i *v56; // [rsp+10h] [rbp-70h]
  unsigned int v57; // [rsp+18h] [rbp-68h]
  const __m128i *v58; // [rsp+18h] [rbp-68h]
  const __m128i *v59; // [rsp+20h] [rbp-60h]
  const void *v60; // [rsp+20h] [rbp-60h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  const void *v62; // [rsp+28h] [rbp-58h]
  int v63; // [rsp+28h] [rbp-58h]
  const __m128i *v64; // [rsp+28h] [rbp-58h]
  __int64 v65; // [rsp+28h] [rbp-58h]
  unsigned int v66; // [rsp+28h] [rbp-58h]
  unsigned __int64 v67; // [rsp+30h] [rbp-50h]
  const __m128i *v68; // [rsp+30h] [rbp-50h]
  const __m128i *v69; // [rsp+30h] [rbp-50h]
  int v70; // [rsp+38h] [rbp-48h] BYREF
  unsigned int v71[3]; // [rsp+3Ch] [rbp-44h] BYREF
  unsigned __int64 v72[7]; // [rsp+48h] [rbp-38h] BYREF

  v6 = a1 + 3552;
  v71[0] = a2;
  v9 = *(_DWORD *)(a1 + 3576);
  v70 = a5;
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 3552);
    v72[0] = 0;
    goto LABEL_49;
  }
  v10 = a5;
  v11 = 1;
  v12 = 0;
  v13 = *(_QWORD *)(a1 + 3560);
  v14 = (v9 - 1) & (37 * a5);
  v15 = v13 + 112LL * v14;
  v16 = *(_DWORD *)v15;
  if ( v10 != *(_DWORD *)v15 )
  {
    while ( v16 != -1 )
    {
      if ( !v12 && v16 == -2 )
        v12 = (int *)v15;
      v14 = (v9 - 1) & (v11 + v14);
      v15 = v13 + 112LL * v14;
      v16 = *(_DWORD *)v15;
      if ( v10 == *(_DWORD *)v15 )
        goto LABEL_3;
      ++v11;
    }
    v48 = *(_DWORD *)(a1 + 3568);
    if ( !v12 )
      v12 = (int *)v15;
    ++*(_QWORD *)(a1 + 3552);
    v49 = v48 + 1;
    v72[0] = (unsigned __int64)v12;
    if ( 4 * v49 < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 3572) - v49 > v9 >> 3 )
      {
LABEL_31:
        *(_DWORD *)(a1 + 3568) = v49;
        if ( *v12 != -1 )
          --*(_DWORD *)(a1 + 3572);
        *v12 = v10;
        v50 = v12 + 6;
        v17 = 0;
        v18 = v12 + 2;
        *v18 = v50;
        v18[1] = 0x100000000LL;
        v19 = 0;
LABEL_4:
        v20 = *v18 + 88 * v17;
        if ( !v20 )
        {
LABEL_8:
          *((_DWORD *)v18 + 2) = v19 + 1;
          goto LABEL_9;
        }
        v21 = *(unsigned int *)(a4 + 8);
        v22 = *(const void **)a4;
        v23 = (void *)(v20 + 16);
        v24 = v71[0];
        *(_QWORD *)(v20 + 8) = 0x100000000LL;
        *(_QWORD *)v20 = v20 + 16;
        v26 = 48 * v21;
        v25 = v26;
        v27 = 0xAAAAAAAAAAAAAAABLL * (v26 >> 4);
        if ( (unsigned __int64)v26 > 0x30 )
        {
          v59 = a3;
          v62 = v22;
          v67 = 0xAAAAAAAAAAAAAAABLL * (v26 >> 4);
          sub_C8D5F0(v20, (const void *)(v20 + 16), v67, 0x30u, (__int64)v22, (__int64)a3);
          LODWORD(v27) = v67;
          v22 = v62;
          a3 = v59;
          v23 = (void *)(*(_QWORD *)v20 + 48LL * *(unsigned int *)(v20 + 8));
        }
        else if ( !v26 )
        {
LABEL_7:
          *(_DWORD *)(v20 + 64) = v24;
          *(_DWORD *)(v20 + 8) = v27 + v26;
          *(__m128i *)(v20 + 72) = _mm_loadu_si128(a3);
          v19 = *((_DWORD *)v18 + 2);
          goto LABEL_8;
        }
        v63 = v27;
        v68 = a3;
        memcpy(v23, v22, v25);
        LODWORD(v26) = *(_DWORD *)(v20 + 8);
        LODWORD(v27) = v63;
        a3 = v68;
        goto LABEL_7;
      }
      v69 = a3;
LABEL_50:
      sub_37C6380(v6, v9);
      sub_37BDBB0(v6, &v70, v72);
      v10 = v70;
      v12 = (int *)v72[0];
      a3 = v69;
      v49 = *(_DWORD *)(a1 + 3568) + 1;
      goto LABEL_31;
    }
LABEL_49:
    v69 = a3;
    v9 *= 2;
    goto LABEL_50;
  }
LABEL_3:
  v17 = *(unsigned int *)(v15 + 16);
  v18 = (_QWORD *)(v15 + 8);
  v19 = v17;
  if ( *(_DWORD *)(v15 + 20) > (unsigned int)v17 )
    goto LABEL_4;
  v64 = a3;
  v37 = sub_C8D7D0(v15 + 8, v15 + 24, 0, 0x58u, v72, (__int64)a3);
  v38 = *(unsigned int *)(v15 + 16);
  v39 = 5 * v38;
  v40 = v37 + 88 * v38;
  if ( v40 )
  {
    v41 = *(unsigned int *)(a4 + 8);
    v35 = *(const void **)a4;
    v42 = (void *)(v40 + 16);
    v34 = 0xAAAAAAAAAAAAAAABLL;
    v43 = v71[0];
    v36 = v64;
    *(_QWORD *)v40 = v40 + 16;
    *(_QWORD *)(v40 + 8) = 0x100000000LL;
    v44 = 48 * v41;
    v39 = v44;
    v45 = 0xAAAAAAAAAAAAAAABLL * (v44 >> 4);
    if ( (unsigned __int64)v44 > 0x30 )
    {
      v55 = v44;
      v56 = v64;
      v57 = v43;
      v60 = v35;
      v65 = v40;
      sub_C8D5F0(v40, (const void *)(v40 + 16), 0xAAAAAAAAAAAAAAABLL * (v44 >> 4), 0x30u, (__int64)v35, (__int64)v36);
      v40 = v65;
      v35 = v60;
      v43 = v57;
      v36 = v56;
      v44 = v55;
      v42 = (void *)(*(_QWORD *)v65 + 48LL * *(unsigned int *)(v65 + 8));
    }
    else if ( !v44 )
    {
LABEL_17:
      *(_DWORD *)(v40 + 64) = v43;
      *(_DWORD *)(v40 + 8) = v39 + v45;
      *(__m128i *)(v40 + 72) = _mm_loadu_si128(v36);
      goto LABEL_18;
    }
    v58 = v36;
    v61 = v40;
    v66 = v43;
    memcpy(v42, v35, v44);
    v40 = v61;
    v36 = v58;
    v43 = v66;
    v39 = *(unsigned int *)(v61 + 8);
    goto LABEL_17;
  }
LABEL_18:
  sub_37BF480(v15 + 8, v37, v39, v34, (__int64)v35, (__int64)v36);
  v46 = v72[0];
  v47 = *(_QWORD *)(v15 + 8);
  if ( v15 + 24 != v47 )
    _libc_free(v47);
  ++*(_DWORD *)(v15 + 16);
  *(_QWORD *)(v15 + 8) = v37;
  *(_DWORD *)(v15 + 20) = v46;
LABEL_9:
  v28 = *(_DWORD *)(a1 + 3608);
  if ( !v28 )
  {
    ++*(_QWORD *)(a1 + 3584);
    v72[0] = 0;
LABEL_44:
    v28 *= 2;
    goto LABEL_45;
  }
  result = v71[0];
  v30 = *(_QWORD *)(a1 + 3592);
  v31 = (v28 - 1) & (37 * v71[0]);
  v32 = (_DWORD *)(v30 + 4LL * v31);
  v33 = *v32;
  if ( *v32 == v71[0] )
    return result;
  v51 = 1;
  v52 = 0;
  while ( v33 != -1 )
  {
    if ( v52 || v33 != -2 )
      v32 = v52;
    v31 = (v28 - 1) & (v51 + v31);
    v33 = *(_DWORD *)(v30 + 4LL * v31);
    if ( v71[0] == v33 )
      return result;
    ++v51;
    v52 = v32;
    v32 = (_DWORD *)(v30 + 4LL * v31);
  }
  v53 = *(_DWORD *)(a1 + 3600);
  if ( !v52 )
    v52 = v32;
  ++*(_QWORD *)(a1 + 3584);
  v54 = v53 + 1;
  v72[0] = (unsigned __int64)v52;
  if ( 4 * v54 >= 3 * v28 )
    goto LABEL_44;
  if ( v28 - *(_DWORD *)(a1 + 3604) - v54 <= v28 >> 3 )
  {
LABEL_45:
    sub_A08C50(a1 + 3584, v28);
    sub_22B31A0(a1 + 3584, (int *)v71, v72);
    result = v71[0];
    v52 = (_DWORD *)v72[0];
    v54 = *(_DWORD *)(a1 + 3600) + 1;
  }
  *(_DWORD *)(a1 + 3600) = v54;
  if ( *v52 != -1 )
    --*(_DWORD *)(a1 + 3604);
  *v52 = result;
  return result;
}
