// Function: sub_33F3AE0
// Address: 0x33f3ae0
//
__int64 *__fastcall sub_33F3AE0(
        _QWORD *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        int a7)
{
  __m128i *v10; // rax
  int v11; // edx
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // rax
  __int64 *v20; // rax
  __int64 *v21; // r12
  unsigned __int64 v23; // r8
  int v24; // r9d
  __int64 v25; // rsi
  unsigned __int8 *v26; // rsi
  __int64 v27; // rcx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  unsigned __int64 v30; // [rsp+8h] [rbp-108h]
  int v31; // [rsp+10h] [rbp-100h]
  __m128i *v32; // [rsp+18h] [rbp-F8h]
  int v33; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v34; // [rsp+20h] [rbp-F0h]
  int v35; // [rsp+20h] [rbp-F0h]
  __int64 *v37; // [rsp+28h] [rbp-E8h]
  __int64 *v38; // [rsp+30h] [rbp-E0h] BYREF
  unsigned __int8 *v39; // [rsp+38h] [rbp-D8h] BYREF
  unsigned __int64 v40[2]; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE *v41; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+58h] [rbp-B8h]
  _BYTE v43[176]; // [rsp+60h] [rbp-B0h] BYREF

  v10 = sub_33ED250((__int64)a1, 1, 0);
  v40[1] = a4;
  v31 = v11;
  v40[0] = a3;
  v42 = 0x2000000000LL;
  v32 = v10;
  v41 = v43;
  sub_33C9670((__int64)&v41, 372, (unsigned __int64)v10, v40, 1, (__int64)&v41);
  v13 = (unsigned int)v42;
  v14 = (unsigned int)v42 + 1LL;
  if ( v14 > HIDWORD(v42) )
  {
    sub_C8D5F0((__int64)&v41, v43, v14, 4u, v12, (__int64)&v41);
    v13 = (unsigned int)v42;
  }
  v15 = HIDWORD(a5);
  *(_DWORD *)&v41[4 * v13] = a5;
  LODWORD(v42) = v42 + 1;
  v16 = (unsigned int)v42;
  if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
  {
    sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 4u, v15, (__int64)&v41);
    v16 = (unsigned int)v42;
    v15 = HIDWORD(a5);
  }
  *(_DWORD *)&v41[4 * v16] = v15;
  LODWORD(v42) = v42 + 1;
  v17 = (unsigned int)v42;
  if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
  {
    sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 4u, v15, (__int64)&v41);
    v17 = (unsigned int)v42;
  }
  *(_DWORD *)&v41[4 * v17] = a6;
  v18 = HIDWORD(a6);
  LODWORD(v42) = v42 + 1;
  v19 = (unsigned int)v42;
  if ( (unsigned __int64)(unsigned int)v42 + 1 > HIDWORD(v42) )
  {
    sub_C8D5F0((__int64)&v41, v43, (unsigned int)v42 + 1LL, 4u, v18, (__int64)&v41);
    v19 = (unsigned int)v42;
    LODWORD(v18) = HIDWORD(a6);
  }
  *(_DWORD *)&v41[4 * v19] = v18;
  LODWORD(v42) = v42 + 1;
  v38 = 0;
  v20 = sub_33CCCF0((__int64)a1, (__int64)&v41, a2, (__int64 *)&v38);
  if ( v20 )
  {
    v21 = v20;
    goto LABEL_11;
  }
  v23 = a1[52];
  v24 = *(_DWORD *)(a2 + 8);
  if ( v23 )
  {
    a1[52] = *(_QWORD *)v23;
LABEL_16:
    v25 = *(_QWORD *)a2;
    v39 = (unsigned __int8 *)v25;
    if ( v25 )
    {
      v30 = v23;
      v33 = v24;
      sub_B96E90((__int64)&v39, v25, 1);
      v23 = v30;
      v24 = v33;
    }
    *(_QWORD *)v23 = 0;
    v26 = v39;
    *(_QWORD *)(v23 + 8) = 0;
    *(_QWORD *)(v23 + 48) = v32;
    *(_QWORD *)(v23 + 16) = 0;
    *(_QWORD *)(v23 + 24) = 372;
    *(_WORD *)(v23 + 34) = -1;
    *(_DWORD *)(v23 + 36) = -1;
    *(_QWORD *)(v23 + 40) = 0;
    *(_QWORD *)(v23 + 56) = 0;
    *(_DWORD *)(v23 + 64) = 0;
    *(_DWORD *)(v23 + 68) = v31;
    *(_DWORD *)(v23 + 72) = v24;
    *(_QWORD *)(v23 + 80) = v26;
    if ( v26 )
    {
      v34 = v23;
      sub_B976B0((__int64)&v39, v26, v23 + 80);
      v23 = v34;
      *(_QWORD *)(v34 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v34 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v23 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v23 + 32) = 0;
    }
    *(_QWORD *)(v23 + 96) = a5;
    *(_QWORD *)(v23 + 104) = a6;
    *(_DWORD *)(v23 + 112) = a7;
    goto LABEL_21;
  }
  v27 = a1[53];
  a1[63] += 120LL;
  v28 = (v27 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v28 + 120 || !v27 )
  {
    v35 = v24;
    v29 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    v24 = v35;
    v23 = v29;
    goto LABEL_16;
  }
  a1[53] = v28 + 120;
  if ( v28 )
  {
    v23 = (v27 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_16;
  }
LABEL_21:
  v37 = (__int64 *)v23;
  sub_33E4EC0((__int64)a1, v23, (__int64)v40, 1);
  sub_C657C0(a1 + 65, v37, v38, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, (__int64)v37);
  v21 = v37;
LABEL_11:
  if ( v41 != v43 )
    _libc_free((unsigned __int64)v41);
  return v21;
}
