// Function: sub_E56990
// Address: 0xe56990
//
__int64 __fastcall sub_E56990(
        __int64 a1,
        __int64 a2,
        int a3,
        char *a4,
        __int64 a5,
        unsigned int a6,
        const char *a7,
        size_t a8,
        __int128 a9,
        char a10,
        char *a11,
        __int64 a12,
        __int64 a13)
{
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // r15
  __int64 v19; // rcx
  __int64 v20; // rdx
  char v21; // dl
  char v22; // al
  unsigned int v23; // ebx
  char v24; // dl
  __int64 v26; // rax
  __int64 v27; // rax
  __m128i v28; // xmm1
  __int64 v29; // rdi
  unsigned int *v30; // rsi
  char v31; // dl
  unsigned int *v32; // rcx
  unsigned int *v33; // rdx
  int v34; // [rsp+48h] [rbp-168h]
  char *v35; // [rsp+50h] [rbp-160h] BYREF
  __int64 v36; // [rsp+58h] [rbp-158h]
  __int64 v37; // [rsp+60h] [rbp-150h] BYREF
  char v38; // [rsp+68h] [rbp-148h]
  _QWORD v39[4]; // [rsp+70h] [rbp-140h] BYREF
  __int16 v40; // [rsp+90h] [rbp-120h]
  _QWORD v41[6]; // [rsp+A0h] [rbp-110h] BYREF
  unsigned int **v42; // [rsp+D0h] [rbp-E0h]
  unsigned int *v43[3]; // [rsp+E0h] [rbp-D0h] BYREF
  _BYTE v44[184]; // [rsp+F8h] [rbp-B8h] BYREF

  v16 = *(_QWORD *)(a2 + 8);
  LODWORD(v41[0]) = a6;
  v35 = a4;
  v17 = *(_QWORD *)(v16 + 1744);
  v36 = a5;
  v18 = v16 + 1736;
  if ( !v17 )
    goto LABEL_13;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v17 + 16);
      v20 = *(_QWORD *)(v17 + 24);
      if ( a6 <= *(_DWORD *)(v17 + 32) )
        break;
      v17 = *(_QWORD *)(v17 + 24);
      if ( !v20 )
        goto LABEL_6;
    }
    v18 = v17;
    v17 = *(_QWORD *)(v17 + 16);
  }
  while ( v19 );
LABEL_6:
  if ( v16 + 1736 == v18 || a6 < *(_DWORD *)(v18 + 32) )
  {
LABEL_13:
    v43[0] = (unsigned int *)v41;
    v26 = sub_E56230((_QWORD *)(v16 + 1728), v18, v43);
    v16 = *(_QWORD *)(a2 + 8);
    v18 = v26;
  }
  v34 = *(_DWORD *)(v18 + 168);
  sub_E798E0(
    (unsigned int)&v37,
    v18 + 40,
    (unsigned int)&v35,
    (unsigned int)&a7,
    *(unsigned __int16 *)(v16 + 1904),
    a3,
    *(_OWORD *)&_mm_loadu_si128((const __m128i *)&a9),
    a10,
    (__int64)a11,
    a12,
    a13);
  v21 = v38 & 1;
  v22 = (2 * (v38 & 1)) | v38 & 0xFD;
  v38 = v22;
  if ( v21 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v38 = v22 & 0xFD;
    v27 = v37;
    v37 = 0;
    *(_QWORD *)a1 = v27 & 0xFFFFFFFFFFFFFFFELL;
LABEL_15:
    if ( v37 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v37 + 8LL))(v37);
    return a1;
  }
  v23 = v37;
  if ( v34 == *(_DWORD *)(v18 + 168) || *(_BYTE *)(*(_QWORD *)(a2 + 312) + 21LL) )
  {
    v24 = *(_BYTE *)(a1 + 8);
    *(_DWORD *)a1 = v37;
    *(_BYTE *)(a1 + 8) = v24 & 0xFC | 2;
    return a1;
  }
  v43[0] = (unsigned int *)v44;
  v41[0] = &unk_49DD288;
  v42 = v43;
  v41[5] = 0x100000000LL;
  v43[1] = 0;
  v43[2] = (unsigned int *)128;
  v41[1] = 2;
  memset(&v41[2], 0, 24);
  sub_CB5980((__int64)v41, 0, 0, 0);
  v28 = _mm_loadu_si128((const __m128i *)&a9);
  sub_E55510(
    a2,
    v23,
    v35,
    v36,
    a7,
    a8,
    v28.m128i_i8[0],
    v28.m128i_i32[2],
    a10,
    a11,
    a12,
    a13,
    *(_BYTE *)(a2 + 747),
    (__int64)v41);
  v29 = *(_QWORD *)(a2 + 16);
  if ( v29 )
  {
    v30 = *v42;
    (*(void (__fastcall **)(__int64, unsigned int *, unsigned int *))(*(_QWORD *)v29 + 40LL))(v29, *v42, v42[1]);
  }
  else
  {
    v30 = (unsigned int *)v39;
    v32 = v42[1];
    v33 = *v42;
    v40 = 261;
    v39[0] = v33;
    v39[1] = v32;
    sub_E99A90(a2, v39);
  }
  v31 = *(_BYTE *)(a1 + 8);
  *(_DWORD *)a1 = v23;
  v41[0] = &unk_49DD388;
  *(_BYTE *)(a1 + 8) = v31 & 0xFC | 2;
  sub_CB5840((__int64)v41);
  if ( (_BYTE *)v43[0] != v44 )
    _libc_free(v43[0], v30);
  if ( (v38 & 2) != 0 )
    sub_9CE230(&v37);
  if ( (v38 & 1) != 0 )
    goto LABEL_15;
  return a1;
}
