// Function: sub_33F1530
// Address: 0x33f1530
//
_QWORD *__fastcall sub_33F1530(_QWORD *a1, __int64 a2)
{
  __m128i *v3; // rax
  __int64 v4; // r9
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v13; // rbx
  __int64 v14; // rax
  unsigned __int8 *v15; // rsi
  __int64 v16; // rcx
  unsigned __int64 v17; // rax
  __int64 *v18; // [rsp+0h] [rbp-D0h] BYREF
  unsigned __int8 *v19; // [rsp+8h] [rbp-C8h] BYREF
  _BYTE *v20; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v21; // [rsp+18h] [rbp-B8h]
  _BYTE v22[176]; // [rsp+20h] [rbp-B0h] BYREF

  v20 = v22;
  v21 = 0x2000000000LL;
  v3 = sub_33ED250((__int64)a1, 1, 0);
  sub_33C9670((__int64)&v20, 324, (unsigned __int64)v3, 0, 0, v4);
  v7 = (unsigned int)v21;
  v8 = (unsigned int)v21 + 1LL;
  if ( v8 > HIDWORD(v21) )
  {
    sub_C8D5F0((__int64)&v20, v22, v8, 4u, v5, v6);
    v7 = (unsigned int)v21;
  }
  *(_DWORD *)&v20[4 * v7] = a2;
  LODWORD(v21) = v21 + 1;
  v9 = (unsigned int)v21;
  if ( (unsigned __int64)(unsigned int)v21 + 1 > HIDWORD(v21) )
  {
    sub_C8D5F0((__int64)&v20, v22, (unsigned int)v21 + 1LL, 4u, v5, v6);
    v9 = (unsigned int)v21;
  }
  *(_DWORD *)&v20[4 * v9] = HIDWORD(a2);
  LODWORD(v21) = v21 + 1;
  v18 = 0;
  v10 = sub_33CCCC0((__int64)a1, (__int64)&v20, (__int64 *)&v18);
  if ( v10 )
  {
    v11 = v10;
    goto LABEL_7;
  }
  v13 = a1[52];
  if ( v13 )
  {
    a1[52] = *(_QWORD *)v13;
LABEL_12:
    v14 = sub_33ECD10(1u);
    v19 = 0;
    *(_QWORD *)v13 = 0;
    v15 = v19;
    *(_QWORD *)(v13 + 48) = v14;
    *(_QWORD *)(v13 + 8) = 0;
    *(_QWORD *)(v13 + 16) = 0;
    *(_QWORD *)(v13 + 24) = 324;
    *(_WORD *)(v13 + 34) = -1;
    *(_DWORD *)(v13 + 36) = -1;
    *(_QWORD *)(v13 + 40) = 0;
    *(_QWORD *)(v13 + 56) = 0;
    *(_QWORD *)(v13 + 64) = 0x100000000LL;
    *(_DWORD *)(v13 + 72) = 0;
    *(_QWORD *)(v13 + 80) = v15;
    if ( v15 )
      sub_B976B0((__int64)&v19, v15, v13 + 80);
    *(_QWORD *)(v13 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v13 + 32) = 0;
    *(_QWORD *)(v13 + 96) = a2;
    goto LABEL_15;
  }
  v16 = a1[53];
  a1[63] += 120LL;
  v17 = (v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v17 + 120 || !v16 )
  {
    v17 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_20;
  }
  a1[53] = v17 + 120;
  if ( v17 )
  {
LABEL_20:
    v13 = v17;
    goto LABEL_12;
  }
LABEL_15:
  v11 = (_QWORD *)v13;
  sub_C657C0(a1 + 65, (__int64 *)v13, v18, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v13);
LABEL_7:
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
  return v11;
}
