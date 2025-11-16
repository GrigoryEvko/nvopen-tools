// Function: sub_2CFFD10
// Address: 0x2cffd10
//
__int64 __fastcall sub_2CFFD10(__int64 a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  char v9; // dl
  __int64 v11; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v12; // [rsp+18h] [rbp-88h]
  int v13; // [rsp+20h] [rbp-80h]
  int v14; // [rsp+24h] [rbp-7Ch]
  int v15; // [rsp+28h] [rbp-78h]
  char v16; // [rsp+2Ch] [rbp-74h]
  _QWORD v17[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v18; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v19; // [rsp+48h] [rbp-58h]
  __int64 v20; // [rsp+50h] [rbp-50h]
  int v21; // [rsp+58h] [rbp-48h]
  char v22; // [rsp+5Ch] [rbp-44h]
  _BYTE v23[64]; // [rsp+60h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v8 = sub_BC1CD0(a4, &unk_4F8FBC8, a3);
  v9 = *a2;
  v11 = v7 + 8;
  v12 = (_QWORD *)(v8 + 8);
  LOBYTE(v13) = v9;
  if ( (_DWORD)qword_5014A48 && (unsigned __int8)sub_2CFEB90((__int64)&v11, a3) )
  {
    v12 = v17;
    v13 = 2;
    v17[0] = &unk_4F82408;
    v15 = 0;
    v16 = 1;
    v18 = 0;
    v19 = v23;
    v20 = 2;
    v21 = 0;
    v22 = 1;
    v14 = 1;
    v11 = 1;
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v17, (__int64)&v11);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v23, (__int64)&v18);
    if ( !v22 )
      _libc_free((unsigned __int64)v19);
    if ( !v16 )
      _libc_free((unsigned __int64)v12);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
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
