// Function: sub_2A88080
// Address: 0x2a88080
//
__int64 __fastcall sub_2A88080(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // [rsp+20h] [rbp-90h] BYREF
  _BYTE *v17; // [rsp+28h] [rbp-88h]
  __int64 v18; // [rsp+30h] [rbp-80h]
  int v19; // [rsp+38h] [rbp-78h]
  char v20; // [rsp+3Ch] [rbp-74h]
  _BYTE v21[16]; // [rsp+40h] [rbp-70h] BYREF
  __int64 v22; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v23; // [rsp+58h] [rbp-58h]
  __int64 v24; // [rsp+60h] [rbp-50h]
  int v25; // [rsp+68h] [rbp-48h]
  char v26; // [rsp+6Ch] [rbp-44h]
  _BYTE v27[64]; // [rsp+70h] [rbp-40h] BYREF

  v7 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  if ( (unsigned __int8)sub_2A87F40(v7 + 8, v8 + 8) )
  {
    v20 = 1;
    v23 = v27;
    v17 = v21;
    v16 = 0;
    v18 = 2;
    v19 = 0;
    v22 = 0;
    v24 = 2;
    v25 = 0;
    v26 = 1;
    sub_2A86990((__int64)&v16, (__int64)&unk_4F875F0, v9, (__int64)v21, v10, a1 + 48);
    sub_2A86990((__int64)&v16, (__int64)&unk_4F81450, v12, v13, v14, v15);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v21, (__int64)&v16);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v27, (__int64)&v22);
    if ( !v26 )
      _libc_free((unsigned __int64)v23);
    if ( !v20 )
      _libc_free((unsigned __int64)v17);
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
