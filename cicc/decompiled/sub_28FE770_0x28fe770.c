// Function: sub_28FE770
// Address: 0x28fe770
//
__int64 __fastcall sub_28FE770(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v6; // ebx
  char v7; // al
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // [rsp+10h] [rbp-90h] BYREF
  int *v17; // [rsp+18h] [rbp-88h]
  __int64 v18; // [rsp+20h] [rbp-80h]
  __int64 v19; // [rsp+28h] [rbp-78h]
  int v20; // [rsp+30h] [rbp-70h] BYREF
  char v21; // [rsp+34h] [rbp-6Ch]
  __int64 v22; // [rsp+40h] [rbp-60h] BYREF
  _BYTE *v23; // [rsp+48h] [rbp-58h]
  __int64 v24; // [rsp+50h] [rbp-50h]
  int v25; // [rsp+58h] [rbp-48h]
  char v26; // [rsp+5Ch] [rbp-44h]
  _BYTE v27[64]; // [rsp+60h] [rbp-40h] BYREF

  v16 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v17 = 0;
  v18 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v19 = 0;
  v20 = 0;
  v21 = 1;
  v6 = sub_F34EF0(a3, (__int64)&v16);
  v7 = sub_28FE0F0(a3, (__int64)&v16);
  v10 = a1 + 32;
  if ( v6 || v7 == 1 )
  {
    v17 = &v20;
    v16 = 0;
    v18 = 2;
    LODWORD(v19) = 0;
    BYTE4(v19) = 1;
    v22 = 0;
    v23 = v27;
    v24 = 2;
    v25 = 0;
    v26 = 1;
    sub_28FE600((__int64)&v16, (__int64)&unk_4F81450, v8, (__int64)&v20, v9, v10);
    sub_28FE600((__int64)&v16, (__int64)&unk_4F875F0, v12, v13, v14, v15);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)&v20, (__int64)&v16);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v27, (__int64)&v22);
    if ( !v26 )
      _libc_free((unsigned __int64)v23);
    if ( !BYTE4(v19) )
      _libc_free((unsigned __int64)v17);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v10;
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
