// Function: sub_A789D0
// Address: 0xa789d0
//
unsigned __int64 __fastcall sub_A789D0(_QWORD *a1, int a2, __int64 a3)
{
  _QWORD *v4; // r14
  _QWORD *v5; // rsi
  __int64 v6; // rax
  __int64 v7; // r9
  int v8; // r8d
  unsigned __int64 v9; // r12
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v16; // [rsp+8h] [rbp-D8h]
  __int64 v17; // [rsp+18h] [rbp-C8h] BYREF
  _QWORD v18[2]; // [rsp+20h] [rbp-C0h] BYREF
  _DWORD v19[44]; // [rsp+30h] [rbp-B0h] BYREF

  v19[0] = a2;
  v4 = (_QWORD *)*a1;
  v18[1] = 0x2000000001LL;
  v18[0] = v19;
  sub_C439F0(a3, v18);
  sub_C439F0(a3 + 16, v18);
  v5 = v18;
  v6 = sub_C65B40(v4 + 50, v18, &v17, off_49D9AB0);
  v7 = a3 + 16;
  v8 = a2;
  v9 = v6;
  if ( !v6 )
  {
    v11 = v4[347];
    v4[357] += 48LL;
    v9 = (v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4[348] >= v9 + 48 && v11 )
    {
      v4[347] = v9 + 48;
    }
    else
    {
      v14 = sub_9D1E70((__int64)(v4 + 347), 48, 48, 3);
      v8 = a2;
      v7 = a3 + 16;
      v9 = v14;
    }
    *(_QWORD *)v9 = 0;
    *(_BYTE *)(v9 + 8) = 4;
    *(_DWORD *)(v9 + 12) = v8;
    v12 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(v9 + 24) = v12;
    if ( v12 > 0x40 )
    {
      v16 = v7;
      sub_C43780(v9 + 16, a3);
      v7 = v16;
    }
    else
    {
      *(_QWORD *)(v9 + 16) = *(_QWORD *)a3;
    }
    v13 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(v9 + 40) = v13;
    if ( v13 > 0x40 )
      sub_C43780(v9 + 32, v7);
    else
      *(_QWORD *)(v9 + 32) = *(_QWORD *)(a3 + 16);
    v5 = (_QWORD *)v9;
    sub_C657C0(v4 + 50, v9, v17, off_49D9AB0);
  }
  if ( (_DWORD *)v18[0] != v19 )
    _libc_free(v18[0], v5);
  return v9;
}
