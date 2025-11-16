// Function: sub_1A81640
// Address: 0x1a81640
//
__int64 __fastcall sub_1A81640(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r12
  __int64 v13; // rbx
  unsigned __int64 v14; // rcx
  double v15; // xmm4_8
  double v16; // xmm5_8
  _QWORD *v17; // rax
  __int64 v18; // rsi
  __int64 *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rdx
  __int64 v22; // r14
  int v23; // r8d
  int v24; // r9d
  __int64 v25; // rbx
  __int64 v26; // r12
  signed __int64 v27; // r12
  _QWORD *v28; // rbx
  _QWORD *v29; // rax
  _BYTE *v30; // rdx
  int v31; // r12d
  __int64 v32; // rcx
  __int64 v33; // rdx
  _QWORD *v34; // r13
  _QWORD *v35; // r12
  unsigned __int64 v36; // rdi
  _BYTE *v38; // [rsp+10h] [rbp-140h] BYREF
  __int64 v39; // [rsp+18h] [rbp-138h]
  _BYTE v40[24]; // [rsp+20h] [rbp-130h] BYREF
  char *v41; // [rsp+38h] [rbp-118h]
  __int64 v42; // [rsp+40h] [rbp-110h]
  char v43; // [rsp+48h] [rbp-108h] BYREF
  __int64 v44; // [rsp+A0h] [rbp-B0h] BYREF
  unsigned int v45; // [rsp+A8h] [rbp-A8h]
  int v46; // [rsp+ACh] [rbp-A4h]
  _QWORD v47[3]; // [rsp+B0h] [rbp-A0h] BYREF
  char v48; // [rsp+C8h] [rbp-88h] BYREF
  __int64 v49; // [rsp+E8h] [rbp-68h]
  char *v50; // [rsp+F0h] [rbp-60h]
  __int64 v51; // [rsp+F8h] [rbp-58h]
  char v52; // [rsp+100h] [rbp-50h] BYREF

  v44 = a1;
  v11 = *(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40);
  if ( (a1 & 0xFFFFFFFFFFFFFFF8LL) != sub_157ED60(v11) )
    return 0;
  v13 = *(_QWORD *)(v11 + 48);
  if ( v13 == v11 + 40 )
    return 0;
  if ( !v13 )
    BUG();
  if ( *(_BYTE *)(v13 - 8) != 77 )
    return 0;
  v14 = sub_1389B50(&v44);
  v17 = (_QWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v44 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF));
  if ( (_QWORD *)v14 == v17 )
    return 0;
  v18 = v13 - 24;
  while ( 1 )
  {
    if ( v18 == *v17 )
    {
      if ( (*(_BYTE *)(v13 - 1) & 0x40) != 0 )
        v19 = *(__int64 **)(v13 - 32);
      else
        v19 = (__int64 *)(v18 - 24LL * (*(_DWORD *)(v13 - 4) & 0xFFFFFFF));
      if ( v19[3 * *(unsigned int *)(v13 + 32) + 1] == v19[3 * *(unsigned int *)(v13 + 32) + 2] )
        return 0;
      v20 = *v19;
      v21 = v19[3];
      if ( v21 != v20 && *(_BYTE *)(v20 + 16) <= 0x10u && *(_BYTE *)(v21 + 16) <= 0x10u )
        break;
    }
    v17 += 3;
    if ( (_QWORD *)v14 == v17 )
      return 0;
  }
  v22 = *(_QWORD *)(*(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 8LL);
  if ( v22 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v22) + 16) - 25) > 9u )
    {
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        goto LABEL_42;
    }
    v25 = v22;
    v26 = 0;
    v38 = v40;
    v39 = 0x200000000LL;
    while ( 1 )
    {
      v25 = *(_QWORD *)(v25 + 8);
      if ( !v25 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v25) + 16) - 25) <= 9u )
      {
        v25 = *(_QWORD *)(v25 + 8);
        ++v26;
        if ( !v25 )
          goto LABEL_22;
      }
    }
LABEL_22:
    v27 = v26 + 1;
    v28 = v40;
    if ( v27 > 2 )
    {
      sub_16CD150((__int64)&v38, v40, v27, 8, v23, v24);
      v28 = &v38[8 * (unsigned int)v39];
    }
    v29 = sub_1648700(v22);
LABEL_27:
    if ( v28 )
      *v28 = v29[5];
    while ( 1 )
    {
      v22 = *(_QWORD *)(v22 + 8);
      if ( !v22 )
        break;
      v29 = sub_1648700(v22);
      if ( (unsigned __int8)(*((_BYTE *)v29 + 16) - 25) <= 9u )
      {
        ++v28;
        goto LABEL_27;
      }
    }
    v30 = v38;
    v31 = v39 + v27;
  }
  else
  {
LABEL_42:
    v31 = 0;
    HIDWORD(v39) = 2;
    v38 = v40;
    v30 = v40;
  }
  LODWORD(v39) = v31;
  v32 = *(_QWORD *)v30;
  v41 = &v43;
  v42 = 0x200000000LL;
  v33 = *((_QWORD *)v30 + 1);
  v47[0] = v32;
  v49 = v33;
  v50 = &v52;
  v47[2] = 0x200000000LL;
  v51 = 0x200000000LL;
  v44 = (__int64)v47;
  v46 = 2;
  v47[1] = &v48;
  v45 = 2;
  sub_1A7F3A0(a1, (__int64)&v44, a2, a3, a4, a5, a6, v15, v16, a9, a10);
  v34 = (_QWORD *)v44;
  v35 = (_QWORD *)(v44 + 56LL * v45);
  if ( (_QWORD *)v44 != v35 )
  {
    do
    {
      v35 -= 7;
      v36 = v35[1];
      if ( (_QWORD *)v36 != v35 + 3 )
        _libc_free(v36);
    }
    while ( v34 != v35 );
    v35 = (_QWORD *)v44;
  }
  if ( v35 != v47 )
    _libc_free((unsigned __int64)v35);
  if ( v38 != v40 )
    _libc_free((unsigned __int64)v38);
  return 1;
}
