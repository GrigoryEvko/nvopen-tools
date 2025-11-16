// Function: sub_2A4B140
// Address: 0x2a4b140
//
void __fastcall sub_2A4B140(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v4; // r13
  unsigned __int64 *v5; // r12
  _QWORD v6[4]; // [rsp+0h] [rbp-690h] BYREF
  unsigned __int64 *v7; // [rsp+20h] [rbp-670h]
  unsigned int v8; // [rsp+28h] [rbp-668h]
  int v9; // [rsp+2Ch] [rbp-664h]
  _QWORD v10[2]; // [rsp+30h] [rbp-660h] BYREF
  char v11; // [rsp+40h] [rbp-650h] BYREF
  __int64 v12; // [rsp+630h] [rbp-60h]
  __int64 v13; // [rsp+638h] [rbp-58h]
  __int64 v14; // [rsp+640h] [rbp-50h]
  unsigned int v15; // [rsp+648h] [rbp-48h]
  __int64 v16; // [rsp+650h] [rbp-40h]
  __int64 v17; // [rsp+658h] [rbp-38h]
  __int64 v18; // [rsp+660h] [rbp-30h]
  __int64 v19; // [rsp+668h] [rbp-28h]

  *(_QWORD *)(a1 + 16) = a1 + 8;
  *(_QWORD *)(a1 + 56) = a1 + 72;
  *(_QWORD *)(a1 + 64) = 0x1400000000LL;
  *(_QWORD *)(a1 + 576) = a1 + 560;
  *(_QWORD *)(a1 + 584) = a1 + 560;
  *(_QWORD *)(a1 + 8) = (a1 + 8) | 4;
  *(_QWORD *)a1 = a2;
  v6[3] = a4;
  v6[1] = a2;
  v6[2] = a3;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_DWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  v6[0] = a1;
  v7 = v10;
  v9 = 32;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v10[0] = &v11;
  v10[1] = 0x400000000LL;
  v8 = 1;
  sub_2A4A620((__int64)v6);
  sub_C7D6A0(v17, 16LL * (unsigned int)v19, 8);
  sub_C7D6A0(v13, 16LL * v15, 8);
  v4 = v7;
  v5 = &v7[6 * v8];
  if ( v7 != v5 )
  {
    do
    {
      v5 -= 6;
      if ( (unsigned __int64 *)*v5 != v5 + 2 )
        _libc_free(*v5);
    }
    while ( v4 != v5 );
    v5 = v7;
  }
  if ( v5 != v10 )
    _libc_free((unsigned __int64)v5);
}
