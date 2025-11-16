// Function: sub_2A411E0
// Address: 0x2a411e0
//
__int64 __fastcall sub_2A411E0(__int64 **a1, __int64 a2, unsigned __int64 a3, unsigned __int64 *a4, __int64 a5)
{
  _BYTE *v7; // r15
  unsigned __int64 *v8; // r12
  __int64 *v9; // r15
  unsigned __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 **v15; // r13
  __int64 v16; // rbx
  _QWORD *v17; // r12
  __int64 v21; // [rsp+28h] [rbp-118h]
  __int64 v22[4]; // [rsp+30h] [rbp-110h] BYREF
  __int16 v23; // [rsp+50h] [rbp-F0h]
  __int64 v24; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v25; // [rsp+68h] [rbp-D8h]
  __int64 v26; // [rsp+70h] [rbp-D0h]
  __int64 v27; // [rsp+78h] [rbp-C8h]
  __int64 *v28; // [rsp+80h] [rbp-C0h]
  __int64 v29; // [rsp+88h] [rbp-B8h]
  _BYTE v30[176]; // [rsp+90h] [rbp-B0h] BYREF

  v24 = 0;
  v25 = 0;
  v7 = sub_BA8CD0((__int64)a1, a2, a3, 0);
  v28 = (__int64 *)v30;
  v26 = 0;
  v27 = 0;
  v29 = 0x1000000000LL;
  sub_2A41150((__int64)v7, (__int64)&v24);
  if ( v7 )
    sub_B30290((__int64)v7);
  v8 = &a4[a5];
  v9 = (__int64 *)sub_BCE3C0(*a1, 0);
  while ( v8 != a4 )
  {
    v10 = *a4++;
    v22[0] = sub_ADB060(v10, (__int64)v9);
    sub_2A40B10((__int64)&v24, v22, v11, v12, v13, v14);
  }
  if ( (_DWORD)v29 )
  {
    v15 = (__int64 **)sub_BCD420(v9, (unsigned int)v29);
    BYTE4(v21) = 0;
    v16 = sub_AD1300(v15, v28, (unsigned int)v29);
    v23 = 261;
    v22[0] = a2;
    v22[1] = a3;
    v17 = sub_BD2C40(88, unk_3F0FAE8);
    if ( v17 )
      sub_B30000((__int64)v17, (__int64)a1, v15, 0, 6, v16, (__int64)v22, 0, 0, v21, 0);
    sub_B31A00((__int64)v17, (__int64)"llvm.metadata", 13);
  }
  if ( v28 != (__int64 *)v30 )
    _libc_free((unsigned __int64)v28);
  return sub_C7D6A0(v25, 8LL * (unsigned int)v27, 8);
}
