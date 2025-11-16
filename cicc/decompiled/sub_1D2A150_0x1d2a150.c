// Function: sub_1D2A150
// Address: 0x1d2a150
//
__int64 __fastcall sub_1D2A150(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        __int64 a4,
        int a5,
        __int64 a6,
        char a7,
        unsigned __int8 a8)
{
  int v10; // ebx
  int v11; // r14d
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r12
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // r9
  __int128 v20; // rdi
  __int64 v21; // rax
  unsigned __int8 *v22; // rsi
  __int64 *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // [rsp+10h] [rbp-F0h]
  int v29; // [rsp+2Ch] [rbp-D4h]
  __int64 *v30; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int8 *v31; // [rsp+38h] [rbp-C8h] BYREF
  unsigned __int64 v32[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v33[176]; // [rsp+50h] [rbp-B0h] BYREF

  v10 = a5;
  v29 = a6;
  if ( !a5 )
  {
    v17 = **(_QWORD **)(a1 + 32) + 112LL;
    if ( (unsigned __int8)sub_1560180(v17, 34) || (unsigned __int8)sub_1560180(v17, 17) )
    {
      v18 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
      v10 = sub_15A9FE0(v18, *a2);
    }
    else
    {
      v24 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
      v10 = sub_15AAE50(v24, *a2);
    }
  }
  v32[0] = (unsigned __int64)v33;
  v11 = a7 == 0 ? 16 : 38;
  v32[1] = 0x2000000000LL;
  v26 = sub_1D29190(a1, a3, a4, a4, (__int64)v33, a6);
  sub_16BD430((__int64)v32, v11);
  sub_16BD4C0((__int64)v32, v26);
  sub_16BD430((__int64)v32, v10);
  sub_16BD3E0((__int64)v32, v29);
  sub_16BD4C0((__int64)v32, (__int64)a2);
  sub_16BD3E0((__int64)v32, a8);
  v30 = 0;
  v12 = sub_1D17910(a1, (__int64)v32, (__int64 *)&v30);
  if ( v12 )
  {
    v15 = (__int64)v12;
  }
  else
  {
    v15 = *(_QWORD *)(a1 + 208);
    v19 = (unsigned __int8)a3;
    if ( v15 )
    {
      *(_QWORD *)(a1 + 208) = *(_QWORD *)v15;
    }
    else
    {
      v25 = sub_145CBF0((__int64 *)(a1 + 216), 112, 8);
      v19 = (unsigned __int8)a3;
      v15 = v25;
    }
    *((_QWORD *)&v20 + 1) = a4;
    *(_QWORD *)&v20 = (unsigned __int8)v19;
    v21 = sub_1D274F0(v20, v13, v14, (__int64)v33, v19);
    *(_WORD *)(v15 + 24) = v11;
    v31 = 0;
    *(_QWORD *)v15 = 0;
    v22 = v31;
    *(_QWORD *)(v15 + 40) = v21;
    *(_QWORD *)(v15 + 56) = 0x100000000LL;
    *(_QWORD *)(v15 + 8) = 0;
    *(_QWORD *)(v15 + 16) = 0;
    *(_DWORD *)(v15 + 28) = -1;
    *(_QWORD *)(v15 + 32) = 0;
    *(_QWORD *)(v15 + 48) = 0;
    *(_DWORD *)(v15 + 64) = 0;
    *(_QWORD *)(v15 + 72) = v22;
    if ( v22 )
      sub_1623210((__int64)&v31, v22, v15 + 72);
    *(_WORD *)(v15 + 80) &= 0xF000u;
    v23 = v30;
    *(_WORD *)(v15 + 26) = 0;
    *(_QWORD *)(v15 + 88) = a2;
    *(_DWORD *)(v15 + 96) = v29;
    *(_DWORD *)(v15 + 100) = v10;
    *(_BYTE *)(v15 + 104) = a8;
    sub_16BDA20((__int64 *)(a1 + 320), (__int64 *)v15, v23);
    sub_1D172A0(a1, v15);
  }
  if ( (_BYTE *)v32[0] != v33 )
    _libc_free(v32[0]);
  return v15;
}
