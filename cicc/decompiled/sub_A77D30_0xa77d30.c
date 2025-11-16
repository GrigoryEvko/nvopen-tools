// Function: sub_A77D30
// Address: 0xa77d30
//
__int64 __fastcall sub_A77D30(__int64 *a1, int a2, __int64 a3)
{
  _QWORD *v4; // rsi
  __int64 v6; // r8
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v10; // rax
  __int64 v11; // [rsp+8h] [rbp-D8h]
  __int64 v12; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v13[2]; // [rsp+20h] [rbp-C0h] BYREF
  int v14; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v15; // [rsp+34h] [rbp-ACh]

  v4 = v13;
  v6 = *a1;
  v15 = a3;
  v13[0] = &v14;
  v7 = v6 + 400;
  v11 = v6;
  v14 = a2;
  v13[1] = 0x2000000003LL;
  v8 = sub_C65B40(v6 + 400, v13, &v12, off_49D9AB0);
  if ( !v8 )
  {
    v10 = sub_A777F0(0x18u, (__int64 *)(v11 + 2640));
    v8 = v10;
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *(_BYTE *)(v10 + 8) = 3;
      *(_DWORD *)(v10 + 12) = a2;
      *(_QWORD *)(v10 + 16) = a3;
    }
    v4 = (_QWORD *)v10;
    sub_C657C0(v7, v10, v12, off_49D9AB0);
  }
  if ( (int *)v13[0] != &v14 )
    _libc_free(v13[0], v4);
  return v8;
}
