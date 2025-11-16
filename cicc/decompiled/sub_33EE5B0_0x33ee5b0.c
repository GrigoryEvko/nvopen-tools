// Function: sub_33EE5B0
// Address: 0x33ee5b0
//
_QWORD *__fastcall sub_33EE5B0(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int16 a5,
        int a6,
        char a7,
        int a8)
{
  __m128i *v12; // rax
  int v13; // ebx
  unsigned int v14; // edx
  __int64 v15; // r9
  __int64 v16; // r9
  __int64 v17; // rax
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rax
  _QWORD *v28; // r12
  char v30; // al
  __int64 *v31; // rdi
  __int64 v32; // rax
  char v33; // al
  __int64 v34; // rax
  unsigned __int64 v35; // r8
  unsigned __int8 *v36; // rsi
  __int64 v37; // rcx
  unsigned __int64 v38; // rax
  __int64 v39; // [rsp+8h] [rbp-F8h]
  unsigned int v40; // [rsp+10h] [rbp-F0h]
  __m128i *v41; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v42; // [rsp+18h] [rbp-E8h]
  char v44; // [rsp+28h] [rbp-D8h]
  __int64 v45; // [rsp+28h] [rbp-D8h]
  __int64 *v46; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v47; // [rsp+38h] [rbp-C8h] BYREF
  _BYTE *v48; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v49; // [rsp+48h] [rbp-B8h]
  _BYTE v50[176]; // [rsp+50h] [rbp-B0h] BYREF

  if ( HIBYTE(a5) )
  {
    v44 = a5;
  }
  else
  {
    v30 = sub_33CC5C0((__int64)a1);
    v31 = (__int64 *)a1[5];
    if ( v30 )
    {
      v34 = sub_2E79000(v31);
      v33 = sub_AE5020(v34, *(_QWORD *)(a2 + 8));
    }
    else
    {
      v32 = sub_2E79000(v31);
      v33 = sub_AE5260(v32, *(_QWORD *)(a2 + 8));
    }
    v44 = v33;
  }
  v12 = sub_33ED250((__int64)a1, a3, a4);
  v13 = a7 == 0 ? 17 : 41;
  v40 = v14;
  v49 = 0x2000000000LL;
  v41 = v12;
  v48 = v50;
  sub_33C9670((__int64)&v48, v13, (unsigned __int64)v12, 0, 0, v15);
  v17 = (unsigned int)v49;
  v18 = 1LL << v44;
  v19 = (unsigned int)v49 + 1LL;
  if ( v19 > HIDWORD(v49) )
  {
    sub_C8D5F0((__int64)&v48, v50, v19, 4u, v18, v16);
    v17 = (unsigned int)v49;
    v18 = 1LL << v44;
  }
  *(_DWORD *)&v48[4 * v17] = v18;
  v20 = HIDWORD(v18);
  LODWORD(v49) = v49 + 1;
  v21 = (unsigned int)v49;
  if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
  {
    v39 = v20;
    sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 4u, v20, v16);
    v21 = (unsigned int)v49;
    v20 = v39;
  }
  *(_DWORD *)&v48[4 * v21] = v20;
  LODWORD(v49) = v49 + 1;
  v22 = (unsigned int)v49;
  if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
  {
    sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 4u, v20, v16);
    v22 = (unsigned int)v49;
  }
  *(_DWORD *)&v48[4 * v22] = a6;
  LODWORD(v49) = v49 + 1;
  v23 = (unsigned int)v49;
  if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
  {
    sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 4u, v20, v16);
    v23 = (unsigned int)v49;
  }
  v24 = HIDWORD(a2);
  *(_DWORD *)&v48[4 * v23] = a2;
  LODWORD(v49) = v49 + 1;
  v25 = (unsigned int)v49;
  if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
  {
    sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 4u, v24, v16);
    v25 = (unsigned int)v49;
    v24 = HIDWORD(a2);
  }
  *(_DWORD *)&v48[4 * v25] = v24;
  LODWORD(v49) = v49 + 1;
  v26 = (unsigned int)v49;
  if ( (unsigned __int64)(unsigned int)v49 + 1 > HIDWORD(v49) )
  {
    sub_C8D5F0((__int64)&v48, v50, (unsigned int)v49 + 1LL, 4u, v24, v16);
    v26 = (unsigned int)v49;
  }
  *(_DWORD *)&v48[4 * v26] = a8;
  LODWORD(v49) = v49 + 1;
  v46 = 0;
  v27 = sub_33CCCC0((__int64)a1, (__int64)&v48, (__int64 *)&v46);
  if ( v27 )
  {
    v28 = v27;
    goto LABEL_17;
  }
  v35 = a1[52];
  if ( v35 )
  {
    a1[52] = *(_QWORD *)v35;
LABEL_26:
    *(_DWORD *)(v35 + 24) = v13;
    v47 = 0;
    *(_QWORD *)(v35 + 48) = v41;
    *(_QWORD *)v35 = 0;
    v36 = v47;
    *(_QWORD *)(v35 + 8) = 0;
    *(_QWORD *)(v35 + 16) = 0;
    *(_DWORD *)(v35 + 28) = 0;
    *(_WORD *)(v35 + 34) = -1;
    *(_DWORD *)(v35 + 36) = -1;
    *(_QWORD *)(v35 + 40) = 0;
    *(_QWORD *)(v35 + 56) = 0;
    *(_DWORD *)(v35 + 64) = 0;
    *(_QWORD *)(v35 + 68) = v40;
    *(_QWORD *)(v35 + 80) = v36;
    if ( v36 )
    {
      v42 = v35;
      sub_B976B0((__int64)&v47, v36, v35 + 80);
      v35 = v42;
      *(_QWORD *)(v42 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v42 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v35 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v35 + 32) = 0;
    }
    *(_QWORD *)(v35 + 96) = a2;
    *(_DWORD *)(v35 + 104) = a6;
    *(_BYTE *)(v35 + 108) = v44;
    *(_DWORD *)(v35 + 112) = a8;
    goto LABEL_29;
  }
  v37 = a1[53];
  a1[63] += 120LL;
  v38 = (v37 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v38 + 120 || !v37 )
  {
    v38 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_34;
  }
  a1[53] = v38 + 120;
  if ( v38 )
  {
LABEL_34:
    v35 = v38;
    goto LABEL_26;
  }
LABEL_29:
  v45 = v35;
  sub_C657C0(a1 + 65, (__int64 *)v35, v46, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v45);
  v28 = (_QWORD *)v45;
LABEL_17:
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v28;
}
