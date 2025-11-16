// Function: sub_22F27B0
// Address: 0x22f27b0
//
__int64 __fastcall sub_22F27B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // r14
  _BYTE *v7; // rdi
  _QWORD v9[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int64 v10; // [rsp+20h] [rbp-90h]
  __int64 v11; // [rsp+28h] [rbp-88h]
  __int64 v12; // [rsp+30h] [rbp-80h]
  __int64 *v13; // [rsp+38h] [rbp-78h]
  __int64 v14; // [rsp+40h] [rbp-70h]
  __int64 v15; // [rsp+48h] [rbp-68h] BYREF
  __int64 v16; // [rsp+50h] [rbp-60h]
  __int64 v17; // [rsp+58h] [rbp-58h]
  __int64 v18; // [rsp+60h] [rbp-50h]
  _BYTE *v19; // [rsp+68h] [rbp-48h]
  __int64 v20; // [rsp+70h] [rbp-40h]
  _BYTE v21[56]; // [rsp+78h] [rbp-38h] BYREF

  v13 = &v15;
  v6 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v19 = v21;
  v9[0] = 0;
  v9[1] = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v20 = 0;
  sub_22F1BF0(v9, a3, v6);
  sub_22EEC20(a3, v6, (__int64)v9);
  v7 = v19;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 32) = &qword_4F82400;
  *(_QWORD *)a1 = 1;
  if ( v7 != v21 )
    _libc_free((unsigned __int64)v7);
  sub_C7D6A0(v16, 8LL * (unsigned int)v18, 8);
  if ( v13 != &v15 )
    _libc_free((unsigned __int64)v13);
  sub_C7D6A0(v10, 8LL * (unsigned int)v12, 8);
  return a1;
}
