// Function: sub_33E5B50
// Address: 0x33e5b50
//
__int64 __fastcall sub_33E5B50(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v8; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // r8
  __int64 *v16; // rbx
  __int64 v17; // r13
  __int64 v19; // rax
  unsigned __int64 *v20; // rbx
  _QWORD *v21; // rcx
  _QWORD *v22; // r12
  __int64 v23; // rdx
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  _QWORD *v30; // [rsp+28h] [rbp-D8h]
  __int64 *v31; // [rsp+38h] [rbp-C8h] BYREF
  _QWORD *v32; // [rsp+40h] [rbp-C0h] BYREF
  int v33; // [rsp+48h] [rbp-B8h]
  _QWORD v34[22]; // [rsp+4Ch] [rbp-B4h] BYREF

  v8 = a1 + 22;
  v34[0] = 0x300000020LL;
  v10 = (unsigned __int16)a2;
  if ( !(_WORD)a2 )
    v10 = a3;
  v34[1] = v10;
  v11 = (unsigned __int16)a4;
  if ( !(_WORD)a4 )
    v11 = a5;
  v12 = (__int64)(a1 + 22);
  v34[2] = v11;
  v13 = (unsigned __int16)a7;
  if ( !(_WORD)a7 )
    v13 = a8;
  v32 = (_QWORD *)((char *)v34 + 4);
  v34[3] = v13;
  v33 = 7;
  v31 = 0;
  v14 = sub_C65B40(v12, (__int64)&v32, (__int64 *)&v31, (__int64)off_4A367B0);
  v15 = a7;
  v16 = v14;
  if ( !v14 )
  {
    v19 = a1[24];
    a1[34] += 48LL;
    v20 = a1 + 24;
    v21 = (_QWORD *)((v19 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    if ( a1[25] >= (unsigned __int64)(v21 + 6) && v19 )
    {
      a1[24] = v21 + 6;
    }
    else
    {
      v26 = sub_9D1E70((__int64)(a1 + 24), 48, 48, 3);
      v15 = a7;
      v21 = (_QWORD *)v26;
    }
    *v21 = a2;
    v21[2] = a4;
    v21[1] = a3;
    v21[4] = v15;
    v21[3] = a5;
    v30 = v21;
    v21[5] = a8;
    v22 = sub_C65D30((__int64)&v32, v20);
    v24 = v23;
    v25 = sub_A777F0(0x28u, (__int64 *)v20);
    v16 = (__int64 *)v25;
    if ( v25 )
    {
      *(_QWORD *)(v25 + 8) = v22;
      *(_QWORD *)v25 = 0;
      *(_QWORD *)(v25 + 16) = v24;
      *(_QWORD *)(v25 + 24) = v30;
      *(_DWORD *)(v25 + 32) = 3;
      *(_DWORD *)(v25 + 36) = sub_939680(v22, (__int64)v22 + 4 * v24);
    }
    sub_C657C0(v8, v16, v31, (__int64)off_4A367B0);
  }
  v17 = v16[3];
  if ( v32 != (_QWORD *)((char *)v34 + 4) )
    _libc_free((unsigned __int64)v32);
  return v17;
}
