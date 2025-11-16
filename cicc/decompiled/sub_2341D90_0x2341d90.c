// Function: sub_2341D90
// Address: 0x2341d90
//
__int64 __fastcall sub_2341D90(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 v4; // rsi
  __int64 v5; // rsi
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  unsigned __int64 v13; // rdi
  _QWORD *v15; // r12
  _QWORD *v16; // r13
  __int64 v17; // rax
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h]
  _QWORD v21[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = a1 + 784;
  v3 = *(_QWORD *)(a1 + 768);
  if ( v3 != v2 )
    _libc_free(v3);
  v4 = *(unsigned int *)(a1 + 752);
  if ( (_DWORD)v4 )
  {
    v15 = *(_QWORD **)(a1 + 736);
    v18 = 0;
    v19 = 0;
    v20 = -4096;
    v16 = &v15[4 * v4];
    v21[0] = 0;
    v21[1] = 0;
    v21[2] = -8192;
    do
    {
      v17 = v15[2];
      if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
        sub_BD60C0(v15);
      v15 += 4;
    }
    while ( v16 != v15 );
    sub_D68D70(v21);
    sub_D68D70(&v18);
    v4 = *(unsigned int *)(a1 + 752);
  }
  v5 = 32 * v4;
  sub_C7D6A0(*(_QWORD *)(a1 + 736), v5, 8);
  v6 = *(_QWORD *)(a1 + 648);
  if ( v6 != a1 + 664 )
    _libc_free(v6);
  v7 = *(_QWORD *)(a1 + 568);
  if ( v7 != a1 + 584 )
    _libc_free(v7);
  if ( (*(_BYTE *)(a1 + 496) & 1) == 0 )
  {
    v5 = 16LL * *(unsigned int *)(a1 + 512);
    sub_C7D6A0(*(_QWORD *)(a1 + 504), v5, 8);
  }
  sub_B72320(a1 + 384, v5);
  v8 = 40LL * *(unsigned int *)(a1 + 376);
  sub_C7D6A0(*(_QWORD *)(a1 + 360), v8, 8);
  sub_278E4C0(a1 + 136, v8, v9, v10, v11, v12, v18, v19, v20);
  v13 = *(_QWORD *)(a1 + 80);
  if ( v13 != a1 + 96 )
    _libc_free(v13);
  return sub_C7D6A0(*(_QWORD *)(a1 + 56), 8LL * *(unsigned int *)(a1 + 72), 8);
}
