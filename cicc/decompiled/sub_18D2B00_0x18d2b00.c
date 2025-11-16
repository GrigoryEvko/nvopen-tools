// Function: sub_18D2B00
// Address: 0x18d2b00
//
__int64 __fastcall sub_18D2B00(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v4; // r14
  __int64 v5; // rax
  _BYTE v6[16]; // [rsp+0h] [rbp-180h] BYREF
  __int64 v7; // [rsp+10h] [rbp-170h]
  char v8; // [rsp+20h] [rbp-160h]
  __int64 v9; // [rsp+30h] [rbp-150h]
  __int64 v10; // [rsp+38h] [rbp-148h]
  __int64 v11; // [rsp+40h] [rbp-140h]
  __int64 v12; // [rsp+48h] [rbp-138h] BYREF
  _BYTE *v13; // [rsp+50h] [rbp-130h]
  _BYTE *v14; // [rsp+58h] [rbp-128h]
  __int64 v15; // [rsp+60h] [rbp-120h]
  int v16; // [rsp+68h] [rbp-118h]
  _BYTE v17[16]; // [rsp+70h] [rbp-110h] BYREF
  __int64 v18; // [rsp+80h] [rbp-100h] BYREF
  _BYTE *v19; // [rsp+88h] [rbp-F8h]
  _BYTE *v20; // [rsp+90h] [rbp-F0h]
  __int64 v21; // [rsp+98h] [rbp-E8h]
  int v22; // [rsp+A0h] [rbp-E0h]
  _BYTE v23[16]; // [rsp+A8h] [rbp-D8h] BYREF
  char v24; // [rsp+B8h] [rbp-C8h]
  __int64 v25; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v26[3]; // [rsp+C8h] [rbp-B8h] BYREF
  _QWORD v27[2]; // [rsp+E0h] [rbp-A0h] BYREF
  unsigned __int64 v28; // [rsp+F0h] [rbp-90h]
  _BYTE v29[16]; // [rsp+108h] [rbp-78h] BYREF
  _QWORD v30[2]; // [rsp+118h] [rbp-68h] BYREF
  unsigned __int64 v31; // [rsp+128h] [rbp-58h]
  _BYTE v32[64]; // [rsp+140h] [rbp-40h] BYREF

  v2 = *a2;
  v26[0] = 0;
  v25 = v2;
  sub_18D2550((__int64)v6, a1, &v25, v26);
  if ( !v8 )
    return *(_QWORD *)(a1 + 32) + 152LL * *(_QWORD *)(v7 + 8) + 8;
  v4 = *(_QWORD *)(a1 + 40) - *(_QWORD *)(a1 + 32);
  v9 = 0;
  *(_QWORD *)(v7 + 8) = 0x86BCA1AF286BCA1BLL * (v4 >> 3);
  v13 = v17;
  v14 = v17;
  v19 = v23;
  v20 = v23;
  v5 = *a2;
  v10 = 0;
  v25 = v5;
  LOWORD(v26[0]) = 0;
  v11 = 0;
  v12 = 0;
  v15 = 2;
  v16 = 0;
  v18 = 0;
  v21 = 2;
  v22 = 0;
  v24 = 0;
  BYTE2(v26[0]) = 0;
  v26[1] = 0;
  v26[2] = 0;
  sub_16CCEE0(v27, (__int64)v29, 2, (__int64)&v12);
  sub_16CCEE0(v30, (__int64)v32, 2, (__int64)&v18);
  v32[16] = v24;
  sub_18D1D70(a1 + 32, (__int64)&v25);
  if ( v31 != v30[1] )
    _libc_free(v31);
  if ( v28 != v27[1] )
    _libc_free(v28);
  if ( v20 != v19 )
    _libc_free((unsigned __int64)v20);
  if ( v14 != v13 )
    _libc_free((unsigned __int64)v14);
  return v4 + *(_QWORD *)(a1 + 32) + 8;
}
