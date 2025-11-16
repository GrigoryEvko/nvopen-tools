// Function: sub_1D29EE0
// Address: 0x1d29ee0
//
_QWORD *__fastcall sub_1D29EE0(_QWORD *a1, int a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 v8; // bl
  int v9; // r12d
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r11
  _QWORD *v15; // r12
  unsigned __int8 v17; // r15
  __int64 v18; // rbx
  __int128 v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // rsi
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v25; // [rsp+10h] [rbp-E0h]
  unsigned __int8 v26; // [rsp+1Ch] [rbp-D4h]
  __int64 *v27; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v28; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v29[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v30[176]; // [rsp+40h] [rbp-B0h] BYREF

  v8 = a3;
  v29[0] = (unsigned __int64)v30;
  v9 = (_BYTE)a5 == 0 ? 15 : 37;
  v26 = a6;
  v29[1] = 0x2000000000LL;
  v25 = sub_1D29190((__int64)a1, a3, a4, (__int64)v30, a5, a6);
  sub_16BD430((__int64)v29, v9);
  sub_16BD4C0((__int64)v29, v25);
  sub_16BD3E0((__int64)v29, a2);
  sub_16BD3E0((__int64)v29, v26);
  v27 = 0;
  v10 = sub_1D17910((__int64)a1, (__int64)v29, (__int64 *)&v27);
  v14 = a4;
  if ( v10 )
  {
    v15 = v10;
  }
  else
  {
    v17 = v8;
    v18 = a1[26];
    if ( v18 )
    {
      a1[26] = *(_QWORD *)v18;
    }
    else
    {
      v23 = sub_145CBF0(a1 + 27, 112, 8);
      v14 = a4;
      v18 = v23;
    }
    *((_QWORD *)&v19 + 1) = v14;
    *(_QWORD *)&v19 = v17;
    v20 = sub_1D274F0(v19, v11, (__int64)v30, v12, v13);
    *(_QWORD *)(v18 + 8) = 0;
    v28 = 0;
    *(_QWORD *)v18 = 0;
    v21 = v28;
    *(_QWORD *)(v18 + 40) = v20;
    *(_QWORD *)(v18 + 16) = 0;
    *(_WORD *)(v18 + 24) = v9;
    *(_DWORD *)(v18 + 28) = -1;
    *(_QWORD *)(v18 + 32) = 0;
    *(_QWORD *)(v18 + 48) = 0;
    *(_QWORD *)(v18 + 56) = 0x100000000LL;
    *(_DWORD *)(v18 + 64) = 0;
    *(_QWORD *)(v18 + 72) = v21;
    if ( v21 )
      sub_1623210((__int64)&v28, v21, v18 + 72);
    *(_WORD *)(v18 + 80) &= 0xF000u;
    v15 = (_QWORD *)v18;
    *(_WORD *)(v18 + 26) = 0;
    *(_DWORD *)(v18 + 84) = a2;
    v22 = v27;
    *(_BYTE *)(v18 + 88) = v26;
    sub_16BDA20(a1 + 40, (__int64 *)v18, v22);
    sub_1D172A0((__int64)a1, v18);
  }
  if ( (_BYTE *)v29[0] != v30 )
    _libc_free(v29[0]);
  return v15;
}
