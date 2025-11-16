// Function: sub_35065A0
// Address: 0x35065a0
//
void __fastcall sub_35065A0(__int64 **a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned __int8 v6; // dl
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // [rsp+0h] [rbp-90h] BYREF
  __int64 v16; // [rsp+8h] [rbp-88h]
  __int64 v17; // [rsp+10h] [rbp-80h]
  unsigned int v18; // [rsp+18h] [rbp-78h]
  unsigned __int64 v19[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE v20[96]; // [rsp+30h] [rbp-60h] BYREF

  sub_3504DF0((__int64)a1);
  v2 = sub_B92180(*a2);
  v6 = *(_BYTE *)(v2 - 16);
  if ( (v6 & 2) != 0 )
  {
    if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v2 - 32) + 40LL) + 32LL) )
      return;
  }
  else if ( !*(_DWORD *)(*(_QWORD *)(v2 - 16 - 8LL * ((v6 >> 2) & 0xF) + 40) + 32LL) )
  {
    return;
  }
  *a1 = a2;
  v19[0] = (unsigned __int64)v20;
  v19[1] = 0x400000000LL;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_3505E20(a1, (__int64)v19, (__int64)&v15, v3, v4, v5);
  v11 = (__int64)a1[28];
  if ( v11 )
  {
    sub_3504900((__int64)a1, v11, v7, v8, v9, v10);
    sub_3504A00((__int64)a1, (__int64)v19, (__int64)&v15, v12, v13, v14);
  }
  sub_C7D6A0(v16, 16LL * v18, 8);
  if ( (_BYTE *)v19[0] != v20 )
    _libc_free(v19[0]);
}
