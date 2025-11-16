// Function: sub_A778C0
// Address: 0xa778c0
//
__int64 __fastcall sub_A778C0(__int64 *a1, int a2, __int64 a3)
{
  _QWORD *v4; // rsi
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // [rsp+0h] [rbp-E0h]
  __int64 v12; // [rsp+8h] [rbp-D8h]
  __int64 v14; // [rsp+18h] [rbp-C8h] BYREF
  int *v15; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v16; // [rsp+28h] [rbp-B8h]
  int v17; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v18; // [rsp+34h] [rbp-ACh]

  v4 = &v15;
  v5 = *a1;
  v16 = 0x2000000001LL;
  v15 = &v17;
  v17 = a2;
  v6 = v5 + 400;
  if ( (unsigned int)(a2 - 86) <= 0xA )
  {
    v18 = a3;
    v11 = v5;
    LODWORD(v16) = 3;
    v7 = sub_C65B40(v5 + 400, &v15, &v14, off_49D9AB0);
    if ( v7 )
      goto LABEL_3;
    v9 = sub_A777F0(0x18u, (__int64 *)(v11 + 2640));
    v7 = v9;
    if ( v9 )
    {
      *(_QWORD *)v9 = 0;
      *(_BYTE *)(v9 + 8) = 1;
      *(_DWORD *)(v9 + 12) = a2;
      *(_QWORD *)(v9 + 16) = a3;
    }
  }
  else
  {
    v12 = v5;
    v7 = sub_C65B40(v5 + 400, &v15, &v14, off_49D9AB0);
    if ( v7 )
      goto LABEL_3;
    v10 = sub_A777F0(0x10u, (__int64 *)(v12 + 2640));
    v7 = v10;
    if ( v10 )
    {
      *(_QWORD *)v10 = 0;
      *(_BYTE *)(v10 + 8) = 0;
      *(_DWORD *)(v10 + 12) = a2;
    }
  }
  v4 = (_QWORD *)v7;
  sub_C657C0(v6, v7, v14, off_49D9AB0);
LABEL_3:
  if ( v15 != &v17 )
    _libc_free(v15, v4);
  return v7;
}
