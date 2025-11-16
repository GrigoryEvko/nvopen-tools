// Function: sub_33EDBD0
// Address: 0x33edbd0
//
_QWORD *__fastcall sub_33EDBD0(_QWORD *a1, int a2, unsigned int a3, __int64 a4, char a5)
{
  int v5; // ebx
  __m128i *v6; // r13
  int v7; // ebx
  unsigned int v8; // edx
  __int64 v9; // r9
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // r12
  unsigned __int64 v17; // r8
  unsigned __int8 *v18; // rsi
  __int64 v19; // rcx
  unsigned __int64 v20; // rax
  unsigned int v21; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v22; // [rsp+0h] [rbp-E0h]
  __int64 v24; // [rsp+8h] [rbp-D8h]
  __int64 *v25; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int8 *v26; // [rsp+18h] [rbp-C8h] BYREF
  _BYTE *v27; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-B8h]
  _BYTE v29[176]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = -(a5 == 0);
  v6 = sub_33ED250((__int64)a1, a3, a4);
  v7 = (v5 & 0xFFFFFFE8) + 39;
  v21 = v8;
  v27 = v29;
  v28 = 0x2000000000LL;
  sub_33C9670((__int64)&v27, v7, (unsigned __int64)v6, 0, 0, v9);
  v12 = (unsigned int)v28;
  v13 = (unsigned int)v28 + 1LL;
  if ( v13 > HIDWORD(v28) )
  {
    sub_C8D5F0((__int64)&v27, v29, v13, 4u, v10, v11);
    v12 = (unsigned int)v28;
  }
  *(_DWORD *)&v27[4 * v12] = a2;
  LODWORD(v28) = v28 + 1;
  v25 = 0;
  v14 = sub_33CCCC0((__int64)a1, (__int64)&v27, (__int64 *)&v25);
  if ( v14 )
  {
    v15 = v14;
    goto LABEL_5;
  }
  v17 = a1[52];
  if ( v17 )
  {
    a1[52] = *(_QWORD *)v17;
LABEL_10:
    *(_DWORD *)(v17 + 24) = v7;
    v26 = 0;
    *(_QWORD *)v17 = 0;
    v18 = v26;
    *(_QWORD *)(v17 + 8) = 0;
    *(_QWORD *)(v17 + 16) = 0;
    *(_DWORD *)(v17 + 28) = 0;
    *(_WORD *)(v17 + 34) = -1;
    *(_DWORD *)(v17 + 36) = -1;
    *(_QWORD *)(v17 + 40) = 0;
    *(_QWORD *)(v17 + 48) = v6;
    *(_QWORD *)(v17 + 56) = 0;
    *(_DWORD *)(v17 + 64) = 0;
    *(_QWORD *)(v17 + 68) = v21;
    *(_QWORD *)(v17 + 80) = v18;
    if ( v18 )
    {
      v22 = v17;
      sub_B976B0((__int64)&v26, v18, v17 + 80);
      v17 = v22;
      *(_QWORD *)(v22 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v22 + 32) = 0;
    }
    else
    {
      *(_QWORD *)(v17 + 88) = 0xFFFFFFFFLL;
      *(_WORD *)(v17 + 32) = 0;
    }
    *(_DWORD *)(v17 + 96) = a2;
    goto LABEL_13;
  }
  v19 = a1[53];
  a1[63] += 120LL;
  v20 = (v19 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a1[54] < v20 + 120 || !v19 )
  {
    v20 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
    goto LABEL_18;
  }
  a1[53] = v20 + 120;
  if ( v20 )
  {
LABEL_18:
    v17 = v20;
    goto LABEL_10;
  }
LABEL_13:
  v24 = v17;
  sub_C657C0(a1 + 65, (__int64 *)v17, v25, (__int64)off_4A367D0);
  sub_33CC420((__int64)a1, v24);
  v15 = (_QWORD *)v24;
LABEL_5:
  if ( v27 != v29 )
    _libc_free((unsigned __int64)v27);
  return v15;
}
