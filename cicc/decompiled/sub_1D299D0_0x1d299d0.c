// Function: sub_1D299D0
// Address: 0x1d299d0
//
_QWORD *__fastcall sub_1D299D0(_QWORD *a1, int a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v7; // bl
  int v8; // r12d
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r12
  unsigned __int8 v15; // r15
  __int64 v16; // rbx
  __int128 v17; // rdi
  __int64 v18; // rax
  unsigned __int8 *v19; // rsi
  __int64 *v20; // rdx
  __int64 v21; // rax
  __int64 v23; // [rsp+18h] [rbp-D8h]
  __int64 *v24; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v25; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v26[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v27[176]; // [rsp+40h] [rbp-B0h] BYREF

  v7 = a3;
  v26[0] = (unsigned __int64)v27;
  v8 = (_BYTE)a5 == 0 ? 14 : 36;
  v26[1] = 0x2000000000LL;
  v23 = sub_1D29190((__int64)a1, a3, a4, (__int64)v27, a5, a4);
  sub_16BD430((__int64)v26, v8);
  sub_16BD4C0((__int64)v26, v23);
  sub_16BD3E0((__int64)v26, a2);
  v24 = 0;
  v9 = sub_1D17910((__int64)a1, (__int64)v26, (__int64 *)&v24);
  v12 = a4;
  if ( v9 )
  {
    v13 = v9;
  }
  else
  {
    v15 = v7;
    v16 = a1[26];
    if ( v16 )
    {
      a1[26] = *(_QWORD *)v16;
    }
    else
    {
      v21 = sub_145CBF0(a1 + 27, 112, 8);
      v12 = a4;
      v16 = v21;
    }
    *((_QWORD *)&v17 + 1) = v12;
    *(_QWORD *)&v17 = v15;
    v18 = sub_1D274F0(v17, v10, (__int64)v27, v11, v12);
    *(_QWORD *)(v16 + 8) = 0;
    v25 = 0;
    *(_QWORD *)v16 = 0;
    v19 = v25;
    *(_QWORD *)(v16 + 40) = v18;
    *(_QWORD *)(v16 + 16) = 0;
    *(_WORD *)(v16 + 24) = v8;
    *(_DWORD *)(v16 + 28) = -1;
    *(_QWORD *)(v16 + 32) = 0;
    *(_QWORD *)(v16 + 48) = 0;
    *(_QWORD *)(v16 + 56) = 0x100000000LL;
    *(_DWORD *)(v16 + 64) = 0;
    *(_QWORD *)(v16 + 72) = v19;
    if ( v19 )
      sub_1623210((__int64)&v25, v19, v16 + 72);
    *(_WORD *)(v16 + 80) &= 0xF000u;
    v13 = (_QWORD *)v16;
    *(_WORD *)(v16 + 26) = 0;
    v20 = v24;
    *(_DWORD *)(v16 + 84) = a2;
    sub_16BDA20(a1 + 40, (__int64 *)v16, v20);
    sub_1D172A0((__int64)a1, v16);
  }
  if ( (_BYTE *)v26[0] != v27 )
    _libc_free(v26[0]);
  return v13;
}
