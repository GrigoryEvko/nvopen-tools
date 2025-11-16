// Function: sub_33F2D30
// Address: 0x33f2d30
//
_QWORD *__fastcall sub_33F2D30(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        int a7,
        int a8)
{
  __m128i *v10; // rax
  int v11; // edx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // r13
  unsigned __int64 v21; // r15
  int v22; // r9d
  __int64 v23; // rsi
  unsigned __int8 *v24; // rsi
  __int64 v25; // rcx
  unsigned __int64 v26; // rax
  __int64 v27; // rax
  int v28; // [rsp+Ch] [rbp-F4h]
  int v29; // [rsp+Ch] [rbp-F4h]
  int v30; // [rsp+10h] [rbp-F0h]
  __m128i *v31; // [rsp+18h] [rbp-E8h]
  __int64 *v32; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int8 *v33; // [rsp+28h] [rbp-D8h] BYREF
  unsigned __int64 v34[2]; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE *v35; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+48h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+50h] [rbp-B0h] BYREF

  v10 = sub_33ED250((__int64)a1, a3, a4);
  v34[1] = a6;
  v30 = v11;
  v34[0] = a5;
  v36 = 0x2000000000LL;
  v31 = v10;
  v35 = v37;
  sub_33C9670((__int64)&v35, 235, (unsigned __int64)v10, v34, 1, v12);
  v15 = (unsigned int)v36;
  v16 = (unsigned int)v36 + 1LL;
  if ( v16 > HIDWORD(v36) )
  {
    sub_C8D5F0((__int64)&v35, v37, v16, 4u, v13, v14);
    v15 = (unsigned int)v36;
  }
  *(_DWORD *)&v35[4 * v15] = a7;
  LODWORD(v36) = v36 + 1;
  v17 = (unsigned int)v36;
  if ( (unsigned __int64)(unsigned int)v36 + 1 > HIDWORD(v36) )
  {
    sub_C8D5F0((__int64)&v35, v37, (unsigned int)v36 + 1LL, 4u, v13, v14);
    v17 = (unsigned int)v36;
  }
  *(_DWORD *)&v35[4 * v17] = a8;
  LODWORD(v36) = v36 + 1;
  v32 = 0;
  v18 = sub_33CCCF0((__int64)a1, (__int64)&v35, a2, (__int64 *)&v32);
  if ( v18 )
  {
    v19 = v18;
    goto LABEL_7;
  }
  v21 = a1[52];
  v22 = *(_DWORD *)(a2 + 8);
  if ( v21 )
  {
    a1[52] = *(_QWORD *)v21;
LABEL_12:
    v23 = *(_QWORD *)a2;
    v33 = (unsigned __int8 *)v23;
    if ( v23 )
    {
      v28 = v22;
      sub_B96E90((__int64)&v33, v23, 1);
      v22 = v28;
    }
    *(_QWORD *)v21 = 0;
    v24 = v33;
    *(_QWORD *)(v21 + 8) = 0;
    *(_QWORD *)(v21 + 48) = v31;
    *(_QWORD *)(v21 + 16) = 0;
    *(_QWORD *)(v21 + 24) = 235;
    *(_WORD *)(v21 + 34) = -1;
    *(_DWORD *)(v21 + 36) = -1;
    *(_QWORD *)(v21 + 40) = 0;
    *(_QWORD *)(v21 + 56) = 0;
    *(_DWORD *)(v21 + 64) = 0;
    *(_DWORD *)(v21 + 68) = v30;
    *(_DWORD *)(v21 + 72) = v22;
    *(_QWORD *)(v21 + 80) = v24;
    if ( v24 )
      sub_B976B0((__int64)&v33, v24, v21 + 80);
    *(_QWORD *)(v21 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v21 + 32) = 0;
    *(_DWORD *)(v21 + 96) = a7;
    *(_DWORD *)(v21 + 100) = a8;
    goto LABEL_17;
  }
  v25 = a1[53];
  a1[63] += 120LL;
  v26 = (v25 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v26 + 120 || !v25 )
  {
    v29 = v22;
    v27 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    v22 = v29;
    v21 = v27;
    goto LABEL_12;
  }
  a1[53] = v26 + 120;
  if ( v26 )
  {
    v21 = (v25 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_12;
  }
LABEL_17:
  sub_33E4EC0((__int64)a1, v21, (__int64)v34, 1);
  v19 = (_QWORD *)v21;
  sub_C657C0(a1 + 65, (__int64 *)v21, v32, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v21);
LABEL_7:
  if ( v35 != v37 )
    _libc_free((unsigned __int64)v35);
  return v19;
}
