// Function: sub_1D25C30
// Address: 0x1d25c30
//
__int64 __fastcall sub_1D25C30(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v4; // r13
  __int64 v5; // rbx
  __int64 v6; // rsi
  _QWORD *v7; // rcx
  __int64 v8; // r13
  __int64 *v10; // r13
  void *v11; // rax
  char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  __int64 *v16; // rdx
  char *srca; // [rsp+0h] [rbp-100h]
  __int64 v19; // [rsp+8h] [rbp-F8h]
  void *v20; // [rsp+18h] [rbp-E8h]
  __int64 v21; // [rsp+18h] [rbp-E8h]
  __int64 *v22; // [rsp+28h] [rbp-D8h] BYREF
  _QWORD v23[2]; // [rsp+30h] [rbp-D0h] BYREF
  unsigned __int64 v24[2]; // [rsp+40h] [rbp-C0h] BYREF
  _BYTE v25[176]; // [rsp+50h] [rbp-B0h] BYREF

  v24[0] = (unsigned __int64)v25;
  v24[1] = 0x2000000000LL;
  sub_16BD430((__int64)v24, a3);
  if ( (_DWORD)a3 )
  {
    v4 = a2;
    v5 = (__int64)&a2[16 * (unsigned int)(a3 - 1) + 16];
    do
    {
      v6 = *v4;
      if ( !(_BYTE)v6 )
        v6 = *((_QWORD *)v4 + 1);
      v4 += 16;
      sub_16BD4D0((__int64)v24, v6);
    }
    while ( (unsigned __int8 *)v5 != v4 );
  }
  v22 = 0;
  v7 = sub_16BDDE0(a1 + 672, (__int64)v24, (__int64 *)&v22);
  if ( !v7 )
  {
    v10 = (__int64 *)(a1 + 544);
    v11 = (void *)sub_145CBF0((__int64 *)(a1 + 544), 16LL * (unsigned int)a3, 8);
    v20 = v11;
    if ( 16 * a3 )
      memcpy(v11, a2, 16 * a3);
    v12 = sub_16BD760((__int64)v24, v10);
    v19 = v13;
    srca = v12;
    v14 = sub_145CBF0(v10, 40, 16);
    *(_QWORD *)v14 = 0;
    *(_QWORD *)(v14 + 8) = srca;
    *(_QWORD *)(v14 + 16) = v19;
    *(_DWORD *)(v14 + 32) = a3;
    *(_QWORD *)(v14 + 24) = v20;
    v23[0] = srca;
    v23[1] = v19;
    v21 = v14;
    v15 = sub_16BDD90((__int64)v23);
    v16 = v22;
    *(_DWORD *)(v21 + 36) = v15;
    sub_16BDA20((__int64 *)(a1 + 672), (__int64 *)v21, v16);
    v7 = (_QWORD *)v21;
  }
  v8 = v7[3];
  if ( (_BYTE *)v24[0] != v25 )
    _libc_free(v24[0]);
  return v8;
}
