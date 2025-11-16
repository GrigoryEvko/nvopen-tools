// Function: sub_A77EC0
// Address: 0xa77ec0
//
unsigned __int64 __fastcall sub_A77EC0(_QWORD *a1, unsigned __int64 *a2, __int64 a3)
{
  _QWORD *v5; // rbx
  _QWORD *v6; // rsi
  __int64 v7; // rax
  _QWORD *v8; // r8
  unsigned __int64 v9; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rax
  _QWORD *v14; // [rsp+8h] [rbp-D8h]
  __int64 v15; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-C0h] BYREF
  _BYTE v17[176]; // [rsp+30h] [rbp-B0h] BYREF

  v5 = (_QWORD *)*a1;
  v16[1] = 0x2000000000LL;
  v16[0] = v17;
  sub_A73D10((__int64)v16, a2, a3);
  v6 = v16;
  v7 = sub_C65B40(v5 + 52, v16, &v15, off_49D9A70);
  v8 = v5 + 52;
  v9 = v7;
  if ( !v7 )
  {
    v11 = v5[330];
    v12 = 8 * a3 + 48;
    v5[340] += v12;
    v9 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v5[331] >= v12 + v9 && v11 )
    {
      v5[330] = v12 + v9;
    }
    else
    {
      v13 = sub_9D1E70((__int64)(v5 + 330), v12, 8 * a3 + 48, 3);
      v8 = v5 + 52;
      v9 = v13;
    }
    v14 = v8;
    sub_A73B90(v9, (__int64 *)a2, a3);
    v6 = (_QWORD *)v9;
    sub_C657C0(v14, v9, v15, off_49D9A70);
  }
  if ( (_BYTE *)v16[0] != v17 )
    _libc_free(v16[0], v6);
  return v9;
}
