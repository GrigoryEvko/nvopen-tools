// Function: sub_2783B40
// Address: 0x2783b40
//
__int64 __fastcall sub_2783B40(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r9
  __int64 v11; // rax
  void *v12; // rsi
  bool v14; // zf
  __int64 v15; // rax
  __int64 v16; // [rsp+0h] [rbp-3A0h]
  __int64 v17; // [rsp+8h] [rbp-398h]
  __int64 v18; // [rsp+8h] [rbp-398h]
  __int64 v19; // [rsp+10h] [rbp-390h]
  __int64 v20; // [rsp+18h] [rbp-388h]
  __int64 v21; // [rsp+20h] [rbp-380h] BYREF
  _QWORD *v22; // [rsp+28h] [rbp-378h]
  int v23; // [rsp+30h] [rbp-370h]
  int v24; // [rsp+34h] [rbp-36Ch]
  int v25; // [rsp+38h] [rbp-368h]
  char v26; // [rsp+3Ch] [rbp-364h]
  _QWORD v27[2]; // [rsp+40h] [rbp-360h] BYREF
  __int64 v28; // [rsp+50h] [rbp-350h] BYREF
  _BYTE *v29; // [rsp+58h] [rbp-348h]
  __int64 v30; // [rsp+60h] [rbp-340h]
  int v31; // [rsp+68h] [rbp-338h]
  char v32; // [rsp+6Ch] [rbp-334h]
  _BYTE v33[16]; // [rsp+70h] [rbp-330h] BYREF
  _BYTE v34[800]; // [rsp+80h] [rbp-320h] BYREF

  v19 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v7 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v20 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v8 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v9 = 0;
  v10 = v8 + 8;
  if ( *a2 )
  {
    v18 = v8 + 8;
    v15 = sub_BC1CD0(a4, &unk_4F8F810, a3);
    v10 = v18;
    v9 = *(_QWORD *)(v15 + 8);
  }
  v16 = v10;
  v17 = v9;
  v11 = sub_B2BEC0(a3);
  sub_2778920((__int64)v34, v11, v19, v7, v20, v16, v17);
  v12 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2780B00((__int64)v34) )
  {
    v14 = *a2 == 0;
    v22 = v27;
    v23 = 2;
    v25 = 0;
    v26 = 1;
    v28 = 0;
    v29 = v33;
    v30 = 2;
    v31 = 0;
    v32 = 1;
    v24 = 1;
    v27[0] = &unk_4F82408;
    v21 = 1;
    if ( !v14 && &unk_4F82408 != (_UNKNOWN *)&qword_4F82400 && &unk_4F82408 != &unk_4F8F810 )
    {
      v24 = 2;
      v21 = 2;
      v27[1] = &unk_4F8F810;
    }
    sub_C8CF70(a1, v12, 2, (__int64)v27, (__int64)&v21);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v33, (__int64)&v28);
    if ( !v32 )
      _libc_free((unsigned __int64)v29);
    if ( !v26 )
      _libc_free((unsigned __int64)v22);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_277A450((__int64)v34);
  return a1;
}
