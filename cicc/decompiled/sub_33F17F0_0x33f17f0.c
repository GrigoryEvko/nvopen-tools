// Function: sub_33F17F0
// Address: 0x33f17f0
//
_QWORD *__fastcall sub_33F17F0(_QWORD *a1, int a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __m128i *v7; // r15
  int v8; // edx
  __int64 v9; // r9
  _QWORD *v10; // rax
  _QWORD *v11; // r12
  __int64 v13; // rsi
  int v14; // r8d
  unsigned __int64 v15; // rbx
  unsigned __int8 *v16; // rsi
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  __int64 v19; // rax
  int v20; // [rsp+0h] [rbp-E0h]
  int v21; // [rsp+8h] [rbp-D8h]
  int v22; // [rsp+8h] [rbp-D8h]
  __int64 *v23; // [rsp+10h] [rbp-D0h] BYREF
  unsigned __int8 *v24; // [rsp+18h] [rbp-C8h] BYREF
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v26[176]; // [rsp+30h] [rbp-B0h] BYREF

  v7 = sub_33ED250((__int64)a1, a4, a5);
  v20 = v8;
  v25[0] = (unsigned __int64)v26;
  v25[1] = 0x2000000000LL;
  sub_33C9670((__int64)v25, a2, (unsigned __int64)v7, 0, 0, v9);
  v23 = 0;
  v10 = sub_33CCCF0((__int64)a1, (__int64)v25, a3, (__int64 *)&v23);
  if ( v10 )
  {
    v11 = v10;
    goto LABEL_3;
  }
  v13 = *(_QWORD *)a3;
  v14 = *(_DWORD *)(a3 + 8);
  v24 = (unsigned __int8 *)v13;
  if ( v13 )
  {
    v21 = v14;
    sub_B96E90((__int64)&v24, v13, 1);
    v14 = v21;
  }
  v15 = a1[52];
  if ( v15 )
  {
    a1[52] = *(_QWORD *)v15;
  }
  else
  {
    v17 = a1[53];
    a1[63] += 120LL;
    v18 = (v17 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( a1[54] >= v18 + 120 && v17 )
    {
      a1[53] = v18 + 120;
      if ( !v18 )
      {
        if ( v24 )
          sub_B91220((__int64)&v24, (__int64)v24);
        goto LABEL_13;
      }
      v15 = (v17 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    }
    else
    {
      v22 = v14;
      v19 = sub_9D1E70((__int64)(a1 + 53), 120, 120, 3);
      v14 = v22;
      v15 = v19;
    }
  }
  *(_QWORD *)v15 = 0;
  v16 = v24;
  *(_QWORD *)(v15 + 8) = 0;
  *(_QWORD *)(v15 + 16) = 0;
  *(_DWORD *)(v15 + 24) = a2;
  *(_DWORD *)(v15 + 28) = 0;
  *(_WORD *)(v15 + 34) = -1;
  *(_DWORD *)(v15 + 36) = -1;
  *(_QWORD *)(v15 + 40) = 0;
  *(_QWORD *)(v15 + 48) = v7;
  *(_QWORD *)(v15 + 56) = 0;
  *(_DWORD *)(v15 + 64) = 0;
  *(_DWORD *)(v15 + 68) = v20;
  *(_DWORD *)(v15 + 72) = v14;
  *(_QWORD *)(v15 + 80) = v16;
  if ( v16 )
    sub_B976B0((__int64)&v24, v16, v15 + 80);
  *(_QWORD *)(v15 + 88) = 0xFFFFFFFFLL;
  *(_WORD *)(v15 + 32) = 0;
LABEL_13:
  sub_C657C0(a1 + 65, (__int64 *)v15, v23, (__int64)off_4A367D0);
  v11 = (_QWORD *)v15;
  sub_33CC420((__int64)a1, v15);
LABEL_3:
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0]);
  return v11;
}
