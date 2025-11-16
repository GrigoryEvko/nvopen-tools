// Function: sub_1D2B300
// Address: 0x1d2b300
//
_QWORD *__fastcall sub_1D2B300(_QWORD *a1, unsigned __int16 a2, __int64 a3, unsigned int a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  _QWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r9
  _QWORD *v11; // r12
  __int64 v13; // rax
  __int64 v14; // rsi
  int v15; // r8d
  __int64 v16; // rcx
  int v17; // edx
  int v18; // r14d
  __int64 v19; // rbx
  unsigned __int8 *v20; // rsi
  __int64 v21; // rax
  __int64 v23; // [rsp+8h] [rbp-E8h]
  __int64 v24; // [rsp+8h] [rbp-E8h]
  int v26; // [rsp+10h] [rbp-E0h]
  int v27; // [rsp+10h] [rbp-E0h]
  __int64 *v29; // [rsp+20h] [rbp-D0h] BYREF
  unsigned __int8 *v30; // [rsp+28h] [rbp-C8h] BYREF
  unsigned __int64 v31[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v32[176]; // [rsp+40h] [rbp-B0h] BYREF

  v31[1] = 0x2000000000LL;
  v31[0] = (unsigned __int64)v32;
  v7 = sub_1D29190((__int64)a1, a4, a5, 0x2000000000LL, a5, a6);
  sub_16BD430((__int64)v31, a2);
  sub_16BD4C0((__int64)v31, v7);
  v29 = 0;
  v8 = sub_1D17920((__int64)a1, (__int64)v31, a3, (__int64 *)&v29);
  if ( v8 )
  {
    v11 = v8;
  }
  else
  {
    v13 = sub_1D29190((__int64)a1, a4, a5, v9, a5, v10);
    v14 = *(_QWORD *)a3;
    v15 = *(_DWORD *)(a3 + 8);
    v16 = v13;
    v18 = v17;
    v30 = (unsigned __int8 *)v14;
    if ( v14 )
    {
      v26 = v15;
      v23 = v13;
      sub_1623A60((__int64)&v30, v14, 2);
      v16 = v23;
      v15 = v26;
    }
    v19 = a1[26];
    if ( v19 )
    {
      a1[26] = *(_QWORD *)v19;
    }
    else
    {
      v24 = v16;
      v27 = v15;
      v21 = sub_145CBF0(a1 + 27, 112, 8);
      v15 = v27;
      v16 = v24;
      v19 = v21;
    }
    *(_QWORD *)v19 = 0;
    v20 = v30;
    *(_QWORD *)(v19 + 8) = 0;
    *(_QWORD *)(v19 + 16) = 0;
    *(_WORD *)(v19 + 24) = a2;
    *(_DWORD *)(v19 + 28) = -1;
    *(_QWORD *)(v19 + 32) = 0;
    *(_QWORD *)(v19 + 40) = v16;
    *(_QWORD *)(v19 + 48) = 0;
    *(_DWORD *)(v19 + 56) = 0;
    *(_DWORD *)(v19 + 60) = v18;
    *(_DWORD *)(v19 + 64) = v15;
    *(_QWORD *)(v19 + 72) = v20;
    if ( v20 )
      sub_1623210((__int64)&v30, v20, v19 + 72);
    *(_WORD *)(v19 + 80) &= 0xF000u;
    v11 = (_QWORD *)v19;
    *(_WORD *)(v19 + 26) = 0;
    sub_16BDA20(a1 + 40, (__int64 *)v19, v29);
    sub_1D172A0((__int64)a1, v19);
  }
  if ( (_BYTE *)v31[0] != v32 )
    _libc_free(v31[0]);
  return v11;
}
