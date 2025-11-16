// Function: sub_2753450
// Address: 0x2753450
//
__int64 __fastcall sub_2753450(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  void *v6; // rsi
  __int64 v8; // [rsp+0h] [rbp-80h] BYREF
  _QWORD *v9; // [rsp+8h] [rbp-78h]
  int v10; // [rsp+10h] [rbp-70h]
  int v11; // [rsp+14h] [rbp-6Ch]
  int v12; // [rsp+18h] [rbp-68h]
  char v13; // [rsp+1Ch] [rbp-64h]
  _QWORD v14[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v15; // [rsp+30h] [rbp-50h] BYREF
  _BYTE *v16; // [rsp+38h] [rbp-48h]
  __int64 v17; // [rsp+40h] [rbp-40h]
  int v18; // [rsp+48h] [rbp-38h]
  char v19; // [rsp+4Ch] [rbp-34h]
  _BYTE v20[48]; // [rsp+50h] [rbp-30h] BYREF

  v5 = sub_BC1CD0(a4, &unk_4F6D3F8, a3);
  v6 = (void *)(a1 + 32);
  if ( (unsigned __int8)sub_2753060(a3, (__int64 *)(v5 + 8)) )
  {
    v9 = v14;
    v10 = 2;
    v14[0] = &unk_4F82408;
    v12 = 0;
    v13 = 1;
    v15 = 0;
    v16 = v20;
    v17 = 2;
    v18 = 0;
    v19 = 1;
    v11 = 1;
    v8 = 1;
    sub_C8CF70(a1, v6, 2, (__int64)v14, (__int64)&v8);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v20, (__int64)&v15);
    if ( !v19 )
      _libc_free((unsigned __int64)v16);
    if ( !v13 )
      _libc_free((unsigned __int64)v9);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v6;
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
  return a1;
}
