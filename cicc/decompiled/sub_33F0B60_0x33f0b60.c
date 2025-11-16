// Function: sub_33F0B60
// Address: 0x33f0b60
//
_QWORD *__fastcall sub_33F0B60(_QWORD *a1, int a2, unsigned int a3, __int64 a4)
{
  __m128i *v6; // r13
  unsigned int v7; // edx
  __int64 v8; // r9
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v16; // rbx
  unsigned __int8 *v17; // rsi
  __int64 v18; // rdi
  __int64 (*v19)(); // r8
  char v20; // al
  __int64 *v21; // rdx
  __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned int v24; // [rsp+8h] [rbp-D8h]
  __int64 *v25; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int8 *v26; // [rsp+18h] [rbp-C8h] BYREF
  _BYTE *v27; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-B8h]
  _BYTE v29[176]; // [rsp+30h] [rbp-B0h] BYREF

  v6 = sub_33ED250((__int64)a1, a3, a4);
  v24 = v7;
  v28 = 0x2000000000LL;
  v27 = v29;
  sub_33C9670((__int64)&v27, 9, (unsigned __int64)v6, 0, 0, v8);
  v11 = (unsigned int)v28;
  v12 = (unsigned int)v28 + 1LL;
  if ( v12 > HIDWORD(v28) )
  {
    sub_C8D5F0((__int64)&v27, v29, v12, 4u, v9, v10);
    v11 = (unsigned int)v28;
  }
  *(_DWORD *)&v27[4 * v11] = a2;
  LODWORD(v28) = v28 + 1;
  v25 = 0;
  v13 = sub_33CCCC0((__int64)a1, (__int64)&v27, (__int64 *)&v25);
  if ( v13 )
  {
    v14 = v13;
    goto LABEL_5;
  }
  v16 = a1[52];
  if ( v16 )
  {
    a1[52] = *(_QWORD *)v16;
LABEL_10:
    v26 = 0;
    *(_QWORD *)v16 = 0;
    v17 = v26;
    *(_QWORD *)(v16 + 8) = 0;
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 24) = 9;
    *(_WORD *)(v16 + 34) = -1;
    *(_DWORD *)(v16 + 36) = -1;
    *(_QWORD *)(v16 + 40) = 0;
    *(_QWORD *)(v16 + 48) = v6;
    *(_QWORD *)(v16 + 56) = 0;
    *(_DWORD *)(v16 + 64) = 0;
    *(_QWORD *)(v16 + 68) = v24;
    *(_QWORD *)(v16 + 80) = v17;
    if ( v17 )
      sub_B976B0((__int64)&v26, v17, v16 + 80);
    *(_QWORD *)(v16 + 88) = 0xFFFFFFFFLL;
    *(_WORD *)(v16 + 32) = 0;
    *(_DWORD *)(v16 + 96) = a2;
    goto LABEL_13;
  }
  v22 = a1[53];
  a1[63] += 120LL;
  v23 = (v22 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v23 + 120 || !v22 )
  {
    v23 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_20;
  }
  a1[53] = v23 + 120;
  if ( v23 )
  {
LABEL_20:
    v16 = v23;
    goto LABEL_10;
  }
LABEL_13:
  v18 = a1[2];
  v19 = *(__int64 (**)())(*(_QWORD *)v18 + 1856LL);
  v20 = 0;
  if ( v19 != sub_302E040 )
    v20 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, _QWORD))v19)(v18, v16, a1[11], a1[10]);
  v14 = (_QWORD *)v16;
  v21 = v25;
  *(_BYTE *)(v16 + 32) = *(_BYTE *)(v16 + 32) & 0xFB | (4 * (v20 & 1));
  sub_C657C0(a1 + 65, (__int64 *)v16, v21, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v16);
LABEL_5:
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v14;
}
