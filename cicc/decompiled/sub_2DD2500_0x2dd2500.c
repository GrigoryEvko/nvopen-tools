// Function: sub_2DD2500
// Address: 0x2dd2500
//
__int64 __fastcall sub_2DD2500(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  void *v4; // r14
  void *v5; // rbx
  __int64 v8; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v9; // [rsp+8h] [rbp-88h]
  int v10; // [rsp+10h] [rbp-80h]
  int v11; // [rsp+14h] [rbp-7Ch]
  int v12; // [rsp+18h] [rbp-78h]
  char v13; // [rsp+1Ch] [rbp-74h]
  _QWORD v14[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v15; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v16; // [rsp+38h] [rbp-58h]
  __int64 v17; // [rsp+40h] [rbp-50h]
  int v18; // [rsp+48h] [rbp-48h]
  char v19; // [rsp+4Ch] [rbp-44h]
  _BYTE v20[64]; // [rsp+50h] [rbp-40h] BYREF

  v4 = (void *)(a1 + 80);
  v5 = (void *)(a1 + 32);
  if ( (*(_BYTE *)(a3 + 3) & 0x40) != 0
    && (sub_BC1CD0(a4, &unk_501DA10, a3), (unsigned __int8)sub_2DD1EE0(a3, (__int64)&unk_501DA10)) )
  {
    v9 = v14;
    v14[0] = &unk_4F81450;
    v10 = 2;
    v12 = 0;
    v13 = 1;
    v15 = 0;
    v16 = v20;
    v17 = 2;
    v18 = 0;
    v19 = 1;
    v11 = 1;
    v8 = 1;
    sub_C8CF70(a1, v5, 2, (__int64)v14, (__int64)&v8);
    sub_C8CF70(a1 + 48, v4, 2, (__int64)v20, (__int64)&v15);
    if ( !v19 )
      _libc_free((unsigned __int64)v16);
    if ( !v13 )
      _libc_free((unsigned __int64)v9);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v5;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v4;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
