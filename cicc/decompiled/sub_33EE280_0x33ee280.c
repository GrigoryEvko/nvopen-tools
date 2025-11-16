// Function: sub_33EE280
// Address: 0x33ee280
//
_QWORD *__fastcall sub_33EE280(_QWORD *a1, int a2, unsigned int a3, __int64 a4, char a5, int a6)
{
  int v6; // ebx
  __m128i *v7; // r14
  int v8; // ebx
  unsigned int v9; // edx
  __int64 v10; // r9
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  unsigned __int64 v19; // r8
  unsigned __int8 *v20; // rsi
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  unsigned int v23; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v24; // [rsp+8h] [rbp-E8h]
  __int64 v27; // [rsp+18h] [rbp-D8h]
  __int64 *v28; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v29; // [rsp+28h] [rbp-C8h] BYREF
  _BYTE *v30; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v31; // [rsp+38h] [rbp-B8h]
  _BYTE v32[176]; // [rsp+40h] [rbp-B0h] BYREF

  v6 = -(a5 == 0);
  v7 = sub_33ED250((__int64)a1, a3, a4);
  v8 = (v6 & 0xFFFFFFE8) + 40;
  v23 = v9;
  v30 = v32;
  v31 = 0x2000000000LL;
  sub_33C9670((__int64)&v30, v8, (unsigned __int64)v7, 0, 0, v10);
  v13 = (unsigned int)v31;
  v14 = (unsigned int)v31 + 1LL;
  if ( v14 > HIDWORD(v31) )
  {
    sub_C8D5F0((__int64)&v30, v32, v14, 4u, v11, v12);
    v13 = (unsigned int)v31;
  }
  *(_DWORD *)&v30[4 * v13] = a2;
  LODWORD(v31) = v31 + 1;
  v15 = (unsigned int)v31;
  if ( (unsigned __int64)(unsigned int)v31 + 1 > HIDWORD(v31) )
  {
    sub_C8D5F0((__int64)&v30, v32, (unsigned int)v31 + 1LL, 4u, v11, v12);
    v15 = (unsigned int)v31;
  }
  *(_DWORD *)&v30[4 * v15] = a6;
  LODWORD(v31) = v31 + 1;
  v28 = 0;
  v16 = sub_33CCCC0((__int64)a1, (__int64)&v30, (__int64 *)&v28);
  if ( v16 )
  {
    v17 = v16;
    goto LABEL_7;
  }
  v19 = a1[52];
  if ( v19 )
  {
    a1[52] = *(_QWORD *)v19;
LABEL_12:
    *(_DWORD *)(v19 + 24) = v8;
    v29 = 0;
    *(_QWORD *)v19 = 0;
    v20 = v29;
    *(_QWORD *)(v19 + 8) = 0;
    *(_QWORD *)(v19 + 16) = 0;
    *(_DWORD *)(v19 + 28) = 0;
    *(_WORD *)(v19 + 34) = -1;
    *(_DWORD *)(v19 + 36) = -1;
    *(_QWORD *)(v19 + 40) = 0;
    *(_QWORD *)(v19 + 48) = v7;
    *(_QWORD *)(v19 + 56) = 0;
    *(_DWORD *)(v19 + 64) = 0;
    *(_QWORD *)(v19 + 68) = v23;
    *(_QWORD *)(v19 + 80) = v20;
    if ( v20 )
    {
      v24 = v19;
      sub_B976B0((__int64)&v29, v20, v19 + 80);
      v19 = v24;
      *(_QWORD *)(v24 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v24 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v19 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v19 + 32) = 0;
    }
    *(_DWORD *)(v19 + 96) = a2;
    *(_DWORD *)(v19 + 100) = a6;
    goto LABEL_15;
  }
  v21 = a1[53];
  a1[63] += 120LL;
  v22 = (v21 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v22 + 120 || !v21 )
  {
    v22 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_20;
  }
  a1[53] = v22 + 120;
  if ( v22 )
  {
LABEL_20:
    v19 = v22;
    goto LABEL_12;
  }
LABEL_15:
  v27 = v19;
  sub_C657C0(a1 + 65, (__int64 *)v19, v28, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v27);
  v17 = (_QWORD *)v27;
LABEL_7:
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return v17;
}
