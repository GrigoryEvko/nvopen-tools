// Function: sub_1D29600
// Address: 0x1d29600
//
__int64 *__fastcall sub_1D29600(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        char a7,
        unsigned __int8 a8)
{
  __int64 v9; // rax
  unsigned int v10; // eax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned int v14; // r15d
  int v15; // r15d
  __int64 *v16; // rax
  char v17; // r8
  __int64 *v18; // r12
  __int64 v20; // r10
  __int64 v21; // r9
  __int64 v22; // rax
  __int128 v23; // [rsp-20h] [rbp-130h]
  __int64 v24; // [rsp+10h] [rbp-100h]
  unsigned int v25; // [rsp+1Ch] [rbp-F4h]
  __int64 *v29; // [rsp+48h] [rbp-C8h] BYREF
  unsigned __int64 v30[2]; // [rsp+50h] [rbp-C0h] BYREF
  _BYTE v31[176]; // [rsp+60h] [rbp-B0h] BYREF

  v9 = sub_1E0A0C0(a1[4]);
  v10 = sub_15A9570(v9, *(_QWORD *)a2);
  if ( v10 <= 0x3F )
  {
    v11 = 64 - v10;
    a6 = a6 << (64 - (unsigned __int8)v10) >> (64 - (unsigned __int8)v10);
  }
  v14 = a7 == 0 ? 0xFFFFFFEA : 0;
  if ( (*(_BYTE *)(a2 + 33) & 0x1C) != 0 )
    v15 = v14 + 35;
  else
    v15 = v14 + 34;
  v30[0] = (unsigned __int64)v31;
  v30[1] = 0x2000000000LL;
  v24 = sub_1D29190((__int64)a1, a4, a5, v11, v12, v13);
  sub_16BD430((__int64)v30, v15);
  sub_16BD4C0((__int64)v30, v24);
  sub_16BD4C0((__int64)v30, a2);
  sub_16BD4D0((__int64)v30, a6);
  sub_16BD3E0((__int64)v30, a8);
  v29 = 0;
  v16 = sub_1D17920((__int64)a1, (__int64)v30, a3, (__int64 *)&v29);
  v17 = a8;
  if ( v16 )
  {
    v18 = v16;
  }
  else
  {
    v18 = (__int64 *)a1[26];
    v20 = a5;
    v21 = (unsigned __int8)a4;
    v25 = *(_DWORD *)(a3 + 8);
    if ( v18 )
    {
      a1[26] = *v18;
    }
    else
    {
      v22 = sub_145CBF0(a1 + 27, 112, 8);
      v17 = a8;
      v21 = (unsigned __int8)a4;
      v20 = a5;
      v18 = (__int64 *)v22;
    }
    *((_QWORD *)&v23 + 1) = v20;
    *(_QWORD *)&v23 = v21;
    sub_1D28EF0((__int64)v18, v15, v25, (__int64 *)a3, a2, a6, v23, v17);
    sub_16BDA20(a1 + 40, v18, v29);
    sub_1D172A0((__int64)a1, (__int64)v18);
  }
  if ( (_BYTE *)v30[0] != v31 )
    _libc_free(v30[0]);
  return v18;
}
