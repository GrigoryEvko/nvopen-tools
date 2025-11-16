// Function: sub_1D2AB00
// Address: 0x1d2ab00
//
_QWORD *__fastcall sub_1D2AB00(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 a7)
{
  unsigned __int8 v9; // bl
  int v10; // r13d
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // r11
  _QWORD *v16; // r12
  unsigned __int8 v18; // r12
  __int64 v19; // rbx
  __int128 v20; // rdi
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v26; // [rsp+18h] [rbp-E8h]
  __int64 *v28; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v29; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v30[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v31[176]; // [rsp+50h] [rbp-B0h] BYREF

  v9 = a3;
  v30[0] = (unsigned __int64)v31;
  v10 = (_BYTE)a6 == 0 ? 18 : 40;
  v30[1] = 0x2000000000LL;
  v26 = sub_1D29190((__int64)a1, a3, a4, 0x2000000000LL, (__int64)v31, a6);
  sub_16BD430((__int64)v30, v10);
  sub_16BD4C0((__int64)v30, v26);
  sub_16BD4C0((__int64)v30, a2);
  sub_16BD4D0((__int64)v30, a5);
  sub_16BD3E0((__int64)v30, a7);
  v28 = 0;
  v11 = sub_1D17910((__int64)a1, (__int64)v30, (__int64 *)&v28);
  v15 = a4;
  if ( v11 )
  {
    v16 = v11;
  }
  else
  {
    v18 = v9;
    v19 = a1[26];
    if ( v19 )
    {
      a1[26] = *(_QWORD *)v19;
    }
    else
    {
      v24 = sub_145CBF0(a1 + 27, 112, 8);
      v15 = a4;
      v19 = v24;
    }
    *((_QWORD *)&v20 + 1) = v15;
    *(_QWORD *)&v20 = v18;
    v21 = sub_1D274F0(v20, v12, v13, (__int64)v31, v14);
    *(_QWORD *)(v19 + 8) = 0;
    v29 = 0;
    *(_QWORD *)v19 = 0;
    v22 = v29;
    *(_QWORD *)(v19 + 40) = v21;
    *(_QWORD *)(v19 + 16) = 0;
    *(_WORD *)(v19 + 24) = v10;
    *(_DWORD *)(v19 + 28) = -1;
    *(_QWORD *)(v19 + 32) = 0;
    *(_QWORD *)(v19 + 48) = 0;
    *(_QWORD *)(v19 + 56) = 0x100000000LL;
    *(_DWORD *)(v19 + 64) = 0;
    *(_QWORD *)(v19 + 72) = v22;
    if ( v22 )
      sub_1623210((__int64)&v29, v22, v19 + 72);
    *(_WORD *)(v19 + 80) &= 0xF000u;
    v16 = (_QWORD *)v19;
    *(_WORD *)(v19 + 26) = 0;
    *(_QWORD *)(v19 + 96) = a5;
    v23 = v28;
    *(_QWORD *)(v19 + 88) = a2;
    *(_BYTE *)(v19 + 104) = a7;
    sub_16BDA20(a1 + 40, (__int64 *)v19, v23);
    sub_1D172A0((__int64)a1, v19);
  }
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v16;
}
