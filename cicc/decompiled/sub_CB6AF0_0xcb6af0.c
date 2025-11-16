// Function: sub_CB6AF0
// Address: 0xcb6af0
//
__int64 __fastcall sub_CB6AF0(__int64 a1, __int64 a2)
{
  char v2; // al
  int v3; // edx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rsi
  unsigned __int64 v7; // rax
  size_t v8; // rdx
  unsigned __int8 *v9; // rsi
  unsigned __int8 *v10; // [rsp+0h] [rbp-90h] BYREF
  size_t v11; // [rsp+8h] [rbp-88h]
  __int64 v12; // [rsp+10h] [rbp-80h]
  _BYTE v13[24]; // [rsp+18h] [rbp-78h] BYREF
  void *v14; // [rsp+30h] [rbp-60h] BYREF
  __int64 v15; // [rsp+38h] [rbp-58h]
  __int64 v16; // [rsp+40h] [rbp-50h]
  __int64 v17; // [rsp+48h] [rbp-48h]
  __int64 v18; // [rsp+50h] [rbp-40h]
  __int64 v19; // [rsp+58h] [rbp-38h]
  unsigned __int8 **v20; // [rsp+60h] [rbp-30h]

  if ( *(_BYTE *)(a2 + 20) )
  {
    v2 = *(_BYTE *)(a2 + 22);
    if ( *(_BYTE *)(a2 + 21) )
      v3 = 2 * (v2 != 0);
    else
      v3 = v2 == 0 ? 1 : 3;
    v4 = *(unsigned int *)(a2 + 16);
    v5 = *(_QWORD *)a2;
    LOBYTE(v15) = 1;
    v14 = (void *)v4;
    sub_C7F500(a1, v5, v3, v4, 1);
    return a1;
  }
  v19 = 0x100000000LL;
  v20 = &v10;
  v14 = &unk_49DD288;
  v10 = v13;
  v11 = 0;
  v12 = 16;
  v15 = 2;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_CB5980((__int64)&v14, 0, 0, 0);
  sub_C7F4E0((__int64)&v14, *(_QWORD *)(a2 + 8), 0, 0);
  v7 = *(unsigned int *)(a2 + 16);
  v8 = v11;
  if ( v7 > v11 )
  {
    sub_CB69B0(a1, v7 - v11);
    v8 = v11;
  }
  v9 = v10;
  sub_CB6200(a1, v10, v8);
  v14 = &unk_49DD388;
  sub_CB5840((__int64)&v14);
  if ( v10 == v13 )
    return a1;
  _libc_free(v10, v9);
  return a1;
}
