// Function: sub_200E9A0
// Address: 0x200e9a0
//
__int64 __fastcall sub_200E9A0(__int64 a1, _QWORD *a2, int a3, __int64 a4, int a5)
{
  __int64 *v6; // r12
  const __m128i *v8; // rax
  int v9; // edi
  unsigned int v10; // r13d
  __m128i *v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rax
  char v14; // si
  __int64 v15; // rax
  __int64 v16; // r10
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // rax
  unsigned int v20; // edx
  unsigned __int8 v21; // al
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rax
  __m128i v29; // xmm2
  __m128i *v30; // rdx
  const __m128i *v31; // rdi
  unsigned int v32; // r13d
  int v33; // eax
  const __m128i *v34; // rsi
  const __m128i *v35; // rax
  void (***v36)(); // rdi
  void (*v37)(); // rax
  __int64 v39; // [rsp+0h] [rbp-FF0h]
  __m128i v41; // [rsp+10h] [rbp-FE0h] BYREF
  int v42; // [rsp+20h] [rbp-FD0h]
  char v43; // [rsp+27h] [rbp-FC9h]
  __int64 v44; // [rsp+28h] [rbp-FC8h]
  _BYTE *v45; // [rsp+30h] [rbp-FC0h]
  _BYTE *v46; // [rsp+38h] [rbp-FB8h]
  _BYTE *v47; // [rsp+40h] [rbp-FB0h]
  __int64 *v48; // [rsp+48h] [rbp-FA8h]
  __int64 v49; // [rsp+50h] [rbp-FA0h]
  __int64 v50; // [rsp+58h] [rbp-F98h]
  __m128i v51; // [rsp+60h] [rbp-F90h]
  __int64 v52; // [rsp+70h] [rbp-F80h] BYREF
  int v53; // [rsp+78h] [rbp-F78h]
  const __m128i *v54; // [rsp+80h] [rbp-F70h] BYREF
  __m128i *v55; // [rsp+88h] [rbp-F68h]
  const __m128i *v56; // [rsp+90h] [rbp-F60h]
  __m128i v57; // [rsp+A0h] [rbp-F50h] BYREF
  __m128i v58; // [rsp+B0h] [rbp-F40h] BYREF
  __int64 v59; // [rsp+C0h] [rbp-F30h]
  __int64 v60; // [rsp+D0h] [rbp-F20h] BYREF
  __int64 v61; // [rsp+D8h] [rbp-F18h]
  __int64 v62; // [rsp+E0h] [rbp-F10h]
  unsigned __int64 v63; // [rsp+E8h] [rbp-F08h]
  __int64 v64; // [rsp+F0h] [rbp-F00h]
  __int64 v65; // [rsp+F8h] [rbp-EF8h]
  __int64 v66; // [rsp+100h] [rbp-EF0h]
  const __m128i *v67; // [rsp+108h] [rbp-EE8h] BYREF
  __m128i *v68; // [rsp+110h] [rbp-EE0h]
  const __m128i *v69; // [rsp+118h] [rbp-ED8h]
  __int64 v70; // [rsp+120h] [rbp-ED0h]
  __int64 v71; // [rsp+128h] [rbp-EC8h] BYREF
  int v72; // [rsp+130h] [rbp-EC0h]
  __int64 v73; // [rsp+138h] [rbp-EB8h]
  _BYTE *v74; // [rsp+140h] [rbp-EB0h]
  __int64 v75; // [rsp+148h] [rbp-EA8h]
  _BYTE v76[1536]; // [rsp+150h] [rbp-EA0h] BYREF
  _BYTE *v77; // [rsp+750h] [rbp-8A0h]
  __int64 v78; // [rsp+758h] [rbp-898h]
  _BYTE v79[512]; // [rsp+760h] [rbp-890h] BYREF
  _BYTE *v80; // [rsp+960h] [rbp-690h]
  __int64 v81; // [rsp+968h] [rbp-688h]
  _BYTE v82[1536]; // [rsp+970h] [rbp-680h] BYREF
  _BYTE *v83; // [rsp+F70h] [rbp-80h]
  __int64 v84; // [rsp+F78h] [rbp-78h]
  _BYTE v85[112]; // [rsp+F80h] [rbp-70h] BYREF

  v6 = &v60;
  v8 = *(const __m128i **)(a4 + 32);
  v44 = a1;
  v9 = *(_DWORD *)(a4 + 56);
  v42 = a5;
  LOBYTE(v46) = a5;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0u;
  v58 = 0u;
  LODWORD(v59) = 0;
  LODWORD(v47) = v9;
  v43 = a5 ^ 1;
  v41 = _mm_loadu_si128(v8);
  if ( v9 != 1 )
  {
    v48 = &v60;
    v10 = 1;
    LOBYTE(v45) = 2 * (v43 & 1);
    while ( 1 )
    {
      v12 = 10LL * v10;
      v13 = *(_QWORD *)(v8->m128i_i64[(unsigned __int64)v12 / 2] + 40) + 16LL * v8->m128i_u32[v12 + 2];
      v14 = *(_BYTE *)v13;
      v61 = *(_QWORD *)(v13 + 8);
      v15 = a2[1];
      LOBYTE(v60) = v14;
      v16 = sub_1F58E60((__int64)v48, *(_QWORD **)(v15 + 48));
      v17 = *(_QWORD *)(a4 + 32);
      v57.m128i_i64[1] = *(_QWORD *)(v17 + v12 * 4);
      LODWORD(v17) = *(_DWORD *)(v17 + v12 * 4 + 8);
      v11 = v55;
      v58.m128i_i64[1] = v16;
      v58.m128i_i32[0] = v17;
      LOBYTE(v59) = (unsigned __int8)v45 | (unsigned __int8)v46 & 1 | v59 & 0xFC;
      if ( v55 == v56 )
      {
        ++v10;
        sub_1D27190(&v54, v55, &v57);
        if ( v10 == (_DWORD)v47 )
          goto LABEL_9;
      }
      else
      {
        if ( v55 )
        {
          *v55 = _mm_loadu_si128(&v57);
          v11[1] = _mm_loadu_si128(&v58);
          v11[2].m128i_i64[0] = v59;
          v11 = v55;
        }
        ++v10;
        v55 = (__m128i *)((char *)v11 + 40);
        if ( v10 == (_DWORD)v47 )
        {
LABEL_9:
          v6 = v48;
          break;
        }
      }
      v8 = *(const __m128i **)(a4 + 32);
    }
  }
  v18 = a2[1];
  v19 = sub_1E0A0C0(*(_QWORD *)(v18 + 32));
  v20 = 8 * sub_15A9520(v19, 0);
  if ( v20 == 32 )
  {
    v21 = 5;
  }
  else if ( v20 > 0x20 )
  {
    v21 = 6;
    if ( v20 != 64 )
    {
      v21 = 0;
      if ( v20 == 128 )
        v21 = 7;
    }
  }
  else
  {
    v21 = 3;
    if ( v20 != 8 )
      v21 = 4 * (v20 == 16);
  }
  v22 = sub_1D27640(v18, *(char **)(*a2 + 8LL * a3 + 74096), v21, 0);
  v23 = *(_QWORD *)(a4 + 40);
  v39 = v24;
  LOBYTE(v24) = *(_BYTE *)v23;
  v61 = *(_QWORD *)(v23 + 8);
  v25 = a2[1];
  LOBYTE(v60) = v24;
  v26 = sub_1F58E60((__int64)v6, *(_QWORD **)(v25 + 48));
  v27 = *(_QWORD *)(a4 + 72);
  v60 = 0;
  v48 = (__int64 *)v26;
  v28 = a2[1];
  v63 = 0xFFFFFFFF00000020LL;
  v70 = v28;
  v46 = v76;
  v74 = v76;
  v75 = 0x2000000000LL;
  v78 = 0x2000000000LL;
  v81 = 0x2000000000LL;
  v47 = v79;
  v77 = v79;
  v45 = v85;
  v83 = v85;
  v61 = 0;
  v62 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v80 = v82;
  v84 = 0x400000000LL;
  v52 = v27;
  if ( v27 )
  {
    sub_1623A60((__int64)&v52, v27, 2);
    v53 = *(_DWORD *)(a4 + 64);
    if ( v71 )
      sub_161E7C0((__int64)&v71, v71);
    v71 = v52;
    if ( v52 )
      sub_1623A60((__int64)&v71, v52, 2);
  }
  else
  {
    v53 = *(_DWORD *)(a4 + 64);
  }
  v29 = _mm_load_si128(&v41);
  v30 = v55;
  v55 = 0;
  v72 = v53;
  v31 = v67;
  v51 = v29;
  v60 = v29.m128i_i64[0];
  LODWORD(v61) = v29.m128i_i32[2];
  v32 = *(_DWORD *)(*a2 + 4LL * a3 + 79648);
  v50 = v39;
  v49 = v22;
  v62 = (__int64)v48;
  v65 = v22;
  LODWORD(v66) = v39;
  LODWORD(v64) = v32;
  v67 = v54;
  v33 = -858993459 * (((char *)v30 - (char *)v54) >> 3);
  v68 = v30;
  v34 = v69;
  v54 = 0;
  HIDWORD(v63) = v33;
  v35 = v56;
  v56 = 0;
  v69 = v35;
  if ( v31 )
    j_j___libc_free_0(v31, (char *)v34 - (char *)v31);
  v36 = *(void (****)())(v70 + 16);
  v37 = **v36;
  if ( v37 != nullsub_684 )
    ((void (__fastcall *)(void (***)(), _QWORD, _QWORD, const __m128i **))v37)(v36, *(_QWORD *)(v70 + 32), v32, &v67);
  LOBYTE(v63) = (2 * (v43 & 1)) | v42 & 1 | v63 & 0xFC;
  if ( v52 )
    sub_161E7C0((__int64)&v52, v52);
  sub_2056920(v44, *a2, v6);
  if ( v83 != v45 )
    _libc_free((unsigned __int64)v83);
  if ( v80 != v82 )
    _libc_free((unsigned __int64)v80);
  if ( v77 != v47 )
    _libc_free((unsigned __int64)v77);
  if ( v74 != v46 )
    _libc_free((unsigned __int64)v74);
  if ( v71 )
    sub_161E7C0((__int64)&v71, v71);
  if ( v67 )
    j_j___libc_free_0(v67, (char *)v69 - (char *)v67);
  if ( v54 )
    j_j___libc_free_0(v54, (char *)v56 - (char *)v54);
  return v44;
}
