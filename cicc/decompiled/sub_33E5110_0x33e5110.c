// Function: sub_33E5110
// Address: 0x33e5110
//
__int64 __fastcall sub_33E5110(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 *v10; // rbx
  __int64 v11; // r13
  __int64 v13; // rax
  unsigned __int64 *v14; // rbx
  _QWORD *v15; // rcx
  _QWORD *v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  _QWORD *v23; // [rsp+18h] [rbp-D8h]
  __int64 *v24; // [rsp+28h] [rbp-C8h] BYREF
  _QWORD *v25; // [rsp+30h] [rbp-C0h] BYREF
  int v26; // [rsp+38h] [rbp-B8h]
  _QWORD v27[22]; // [rsp+3Ch] [rbp-B4h] BYREF

  v27[0] = 0x200000020LL;
  v6 = (unsigned __int16)a2;
  if ( !(_WORD)a2 )
    v6 = a3;
  v27[1] = v6;
  v7 = (unsigned __int16)a4;
  if ( !(_WORD)a4 )
    v7 = a5;
  v25 = (_QWORD *)((char *)v27 + 4);
  v27[2] = v7;
  v26 = 5;
  v24 = 0;
  v8 = sub_C65B40((__int64)(a1 + 22), (__int64)&v25, (__int64 *)&v24, (__int64)off_4A367B0);
  v9 = a2;
  v10 = v8;
  if ( !v8 )
  {
    v13 = a1[24];
    a1[34] += 32;
    v14 = (unsigned __int64 *)(a1 + 24);
    v15 = (_QWORD *)((v13 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( a1[25] >= (unsigned __int64)(v15 + 4) && v13 )
    {
      a1[24] = (__int64)(v15 + 4);
    }
    else
    {
      v20 = sub_9D1E70((__int64)(a1 + 24), 32, 32, 3);
      v9 = a2;
      v15 = (_QWORD *)v20;
    }
    *v15 = v9;
    v15[2] = a4;
    v15[1] = a3;
    v23 = v15;
    v15[3] = a5;
    v16 = sub_C65D30((__int64)&v25, v14);
    v18 = v17;
    v19 = sub_A777F0(0x28u, (__int64 *)v14);
    v10 = (__int64 *)v19;
    if ( v19 )
    {
      *(_QWORD *)(v19 + 8) = v16;
      *(_QWORD *)v19 = 0;
      *(_QWORD *)(v19 + 16) = v18;
      *(_QWORD *)(v19 + 24) = v23;
      *(_DWORD *)(v19 + 32) = 2;
      *(_DWORD *)(v19 + 36) = sub_939680(v16, (__int64)v16 + 4 * v18);
    }
    sub_C657C0(a1 + 22, v10, v24, (__int64)off_4A367B0);
  }
  v11 = v10[3];
  if ( v25 != (_QWORD *)((char *)v27 + 4) )
    _libc_free((unsigned __int64)v25);
  return v11;
}
