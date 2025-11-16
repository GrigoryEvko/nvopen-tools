// Function: sub_33F29A0
// Address: 0x33f29a0
//
_QWORD *__fastcall sub_33F29A0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        unsigned __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6)
{
  __m128i *v7; // rax
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  __int64 v18; // r13
  __int64 v19; // rcx
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  int v24; // [rsp+8h] [rbp-F8h]
  __int64 v26; // [rsp+10h] [rbp-F0h]
  __int64 *v28; // [rsp+20h] [rbp-E0h] BYREF
  unsigned __int8 *v29; // [rsp+28h] [rbp-D8h] BYREF
  unsigned __int64 v30[2]; // [rsp+30h] [rbp-D0h] BYREF
  _BYTE *v31; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v32; // [rsp+48h] [rbp-B8h]
  _BYTE v33[176]; // [rsp+50h] [rbp-B0h] BYREF

  v30[0] = a4;
  v30[1] = a5;
  v31 = v33;
  v32 = 0x2000000000LL;
  v7 = sub_33ED250((__int64)a1, 1, 0);
  sub_33C9670((__int64)&v31, a2, (unsigned __int64)v7, v30, 1, v8);
  v11 = (unsigned int)v32;
  v12 = (unsigned int)v32 + 1LL;
  if ( v12 > HIDWORD(v32) )
  {
    sub_C8D5F0((__int64)&v31, v33, v12, 4u, v9, v10);
    v11 = (unsigned int)v32;
  }
  v13 = HIDWORD(a6);
  *(_DWORD *)&v31[4 * v11] = a6;
  LODWORD(v32) = v32 + 1;
  v14 = (unsigned int)v32;
  if ( (unsigned __int64)(unsigned int)v32 + 1 > HIDWORD(v32) )
  {
    sub_C8D5F0((__int64)&v31, v33, (unsigned int)v32 + 1LL, 4u, v13, v10);
    v14 = (unsigned int)v32;
    v13 = HIDWORD(a6);
  }
  *(_DWORD *)&v31[4 * v14] = v13;
  LODWORD(v32) = v32 + 1;
  v28 = 0;
  v15 = sub_33CCCC0((__int64)a1, (__int64)&v31, (__int64 *)&v28);
  if ( v15 )
  {
    v16 = v15;
    goto LABEL_7;
  }
  v18 = a1[52];
  v24 = *(_DWORD *)(a3 + 8);
  if ( v18 )
  {
    a1[52] = *(_QWORD *)v18;
LABEL_12:
    v19 = sub_33ECD10(1u);
    v20 = *(_QWORD *)a3;
    v29 = (unsigned __int8 *)v20;
    if ( v20 )
    {
      v26 = v19;
      sub_B96E90((__int64)&v29, v20, 1);
      v19 = v26;
    }
    *(_QWORD *)v18 = 0;
    *(_WORD *)(v18 + 34) = -1;
    v21 = v29;
    *(_DWORD *)(v18 + 24) = a2;
    *(_QWORD *)(v18 + 64) = 0x100000000LL;
    *(_QWORD *)(v18 + 8) = 0;
    *(_QWORD *)(v18 + 16) = 0;
    *(_DWORD *)(v18 + 28) = 0;
    *(_DWORD *)(v18 + 36) = -1;
    *(_QWORD *)(v18 + 40) = 0;
    *(_QWORD *)(v18 + 48) = v19;
    *(_QWORD *)(v18 + 56) = 0;
    *(_DWORD *)(v18 + 72) = v24;
    *(_QWORD *)(v18 + 80) = v21;
    if ( v21 )
      sub_B976B0((__int64)&v29, v21, v18 + 80);
    *(_QWORD *)(v18 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v18 + 32) = 0;
    *(_QWORD *)(v18 + 96) = a6;
    goto LABEL_17;
  }
  v22 = a1[53];
  a1[63] += 120LL;
  v23 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v23 + 120 || !v22 )
  {
    v23 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_22;
  }
  a1[53] = v23 + 120;
  if ( v23 )
  {
LABEL_22:
    v18 = v23;
    goto LABEL_12;
  }
LABEL_17:
  sub_33E4EC0((__int64)a1, v18, (__int64)v30, 1);
  sub_C657C0(a1 + 65, (__int64 *)v18, v28, (__int64)off_4A367D0);
  v16 = (_QWORD *)v18;
  sub_33CC420((__int64)a1, v18);
LABEL_7:
  if ( v31 != v33 )
    _libc_free((unsigned __int64)v31);
  return v16;
}
