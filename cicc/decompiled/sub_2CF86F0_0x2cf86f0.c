// Function: sub_2CF86F0
// Address: 0x2cf86f0
//
__int64 __fastcall sub_2CF86F0(__int64 a1, __int64 a2, size_t a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v6; // r14
  void *v7; // r13
  __int64 v9; // [rsp+0h] [rbp-90h] BYREF
  _QWORD *v10; // [rsp+8h] [rbp-88h]
  int v11; // [rsp+10h] [rbp-80h]
  int v12; // [rsp+14h] [rbp-7Ch]
  int v13; // [rsp+18h] [rbp-78h]
  char v14; // [rsp+1Ch] [rbp-74h]
  _QWORD v15[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v16; // [rsp+30h] [rbp-60h] BYREF
  _BYTE *v17; // [rsp+38h] [rbp-58h]
  __int64 v18; // [rsp+40h] [rbp-50h]
  int v19; // [rsp+48h] [rbp-48h]
  char v20; // [rsp+4Ch] [rbp-44h]
  _BYTE v21[64]; // [rsp+50h] [rbp-40h] BYREF

  v6 = (void *)(a1 + 32);
  v7 = (void *)(a1 + 80);
  if ( (_DWORD)qword_50147A8 && sub_2CF79F0(a3, a2, a3, a4, a5, a6) )
  {
    v10 = v15;
    v15[0] = &unk_4F82408;
    v11 = 2;
    v13 = 0;
    v14 = 1;
    v16 = 0;
    v17 = v21;
    v18 = 2;
    v19 = 0;
    v20 = 1;
    v12 = 1;
    v9 = 1;
    sub_C8CF70(a1, v6, 2, (__int64)v15, (__int64)&v9);
    sub_C8CF70(a1 + 48, v7, 2, (__int64)v21, (__int64)&v16);
    if ( !v20 )
      _libc_free((unsigned __int64)v17);
    if ( !v14 )
      _libc_free((unsigned __int64)v10);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v6;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v7;
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
