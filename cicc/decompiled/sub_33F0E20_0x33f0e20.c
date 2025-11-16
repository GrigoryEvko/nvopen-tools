// Function: sub_33F0E20
// Address: 0x33f0e20
//
_QWORD *__fastcall sub_33F0E20(
        _QWORD *a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        char a6,
        int a7)
{
  int v8; // ebx
  __m128i *v9; // rax
  int v10; // ebx
  unsigned int v11; // edx
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  _QWORD *v23; // rax
  _QWORD *v24; // r12
  unsigned __int64 v26; // r8
  unsigned __int8 *v27; // rsi
  __int64 v28; // rcx
  unsigned __int64 v29; // rax
  unsigned int v30; // [rsp+8h] [rbp-E8h]
  __m128i *v31; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v32; // [rsp+10h] [rbp-E0h]
  __int64 v34; // [rsp+18h] [rbp-D8h]
  __int64 *v35; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v36; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v37; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-B8h]
  _BYTE v39[176]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = -(a6 == 0);
  v9 = sub_33ED250((__int64)a1, a3, a4);
  v10 = (v8 & 0xFFFFFFE8) + 43;
  v30 = v11;
  v38 = 0x2000000000LL;
  v31 = v9;
  v37 = v39;
  sub_33C9670((__int64)&v37, v10, (unsigned __int64)v9, 0, 0, v12);
  v15 = (unsigned int)v38;
  v16 = (unsigned int)v38 + 1LL;
  if ( v16 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, v16, 4u, v13, v14);
    v15 = (unsigned int)v38;
  }
  v17 = HIDWORD(a2);
  *(_DWORD *)&v37[4 * v15] = a2;
  LODWORD(v38) = v38 + 1;
  v18 = (unsigned int)v38;
  if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 4u, v17, v14);
    v18 = (unsigned int)v38;
    v17 = HIDWORD(a2);
  }
  *(_DWORD *)&v37[4 * v18] = v17;
  LODWORD(v38) = v38 + 1;
  v19 = (unsigned int)v38;
  if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 4u, v17, v14);
    v19 = (unsigned int)v38;
  }
  *(_DWORD *)&v37[4 * v19] = a5;
  v20 = HIDWORD(a5);
  LODWORD(v38) = v38 + 1;
  v21 = (unsigned int)v38;
  if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 4u, v20, v14);
    v21 = (unsigned int)v38;
    v20 = HIDWORD(a5);
  }
  *(_DWORD *)&v37[4 * v21] = v20;
  LODWORD(v38) = v38 + 1;
  v22 = (unsigned int)v38;
  if ( (unsigned __int64)(unsigned int)v38 + 1 > HIDWORD(v38) )
  {
    sub_C8D5F0((__int64)&v37, v39, (unsigned int)v38 + 1LL, 4u, v20, v14);
    v22 = (unsigned int)v38;
  }
  *(_DWORD *)&v37[4 * v22] = a7;
  LODWORD(v38) = v38 + 1;
  v35 = 0;
  v23 = sub_33CCCC0((__int64)a1, (__int64)&v37, (__int64 *)&v35);
  if ( v23 )
  {
    v24 = v23;
    goto LABEL_13;
  }
  v26 = a1[52];
  if ( v26 )
  {
    a1[52] = *(_QWORD *)v26;
LABEL_18:
    *(_DWORD *)(v26 + 24) = v10;
    v36 = 0;
    *(_QWORD *)(v26 + 48) = v31;
    *(_QWORD *)v26 = 0;
    v27 = v36;
    *(_QWORD *)(v26 + 8) = 0;
    *(_QWORD *)(v26 + 16) = 0;
    *(_DWORD *)(v26 + 28) = 0;
    *(_WORD *)(v26 + 34) = -1;
    *(_DWORD *)(v26 + 36) = -1;
    *(_QWORD *)(v26 + 40) = 0;
    *(_QWORD *)(v26 + 56) = 0;
    *(_DWORD *)(v26 + 64) = 0;
    *(_QWORD *)(v26 + 68) = v30;
    *(_QWORD *)(v26 + 80) = v27;
    if ( v27 )
    {
      v32 = v26;
      sub_B976B0((__int64)&v36, v27, v26 + 80);
      v26 = v32;
      *(_QWORD *)(v32 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v32 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v26 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v26 + 32) = 0;
    }
    *(_QWORD *)(v26 + 96) = a2;
    *(_QWORD *)(v26 + 104) = a5;
    *(_DWORD *)(v26 + 112) = a7;
    goto LABEL_21;
  }
  v28 = a1[53];
  a1[63] += 120LL;
  v29 = (v28 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v29 + 120 || !v28 )
  {
    v29 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_26;
  }
  a1[53] = v29 + 120;
  if ( v29 )
  {
LABEL_26:
    v26 = v29;
    goto LABEL_18;
  }
LABEL_21:
  v34 = v26;
  sub_C657C0(a1 + 65, (__int64 *)v26, v35, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v34);
  v24 = (_QWORD *)v34;
LABEL_13:
  if ( v37 != v39 )
    _libc_free((unsigned __int64)v37);
  return v24;
}
