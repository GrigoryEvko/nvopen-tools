// Function: sub_278E8B0
// Address: 0x278e8b0
//
__int64 __fastcall sub_278E8B0(__int64 a1)
{
  unsigned __int64 v2; // rdi
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // rdi
  __int64 *v6; // r14
  __int64 *v7; // rbx
  __int64 i; // rax
  __int64 v9; // rdi
  unsigned int v10; // ecx
  __int64 v11; // rsi
  __int64 *v12; // rbx
  unsigned __int64 v13; // r13
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // r13
  __int64 v21; // rax
  _QWORD v22[2]; // [rsp+0h] [rbp-60h] BYREF
  __int64 v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+20h] [rbp-40h]
  __int64 v25; // [rsp+28h] [rbp-38h]
  __int64 v26; // [rsp+30h] [rbp-30h]

  *(_QWORD *)a1 = off_4A20CB0;
  v2 = *(_QWORD *)(a1 + 944);
  if ( v2 != a1 + 960 )
    _libc_free(v2);
  v3 = *(unsigned int *)(a1 + 928);
  if ( (_DWORD)v3 )
  {
    v19 = *(_QWORD **)(a1 + 912);
    v22[0] = 0;
    v22[1] = 0;
    v23 = -4096;
    v20 = &v19[4 * v3];
    v24 = 0;
    v25 = 0;
    v26 = -8192;
    do
    {
      v21 = v19[2];
      if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        sub_BD60C0(v19);
      v19 += 4;
    }
    while ( v20 != v19 );
    if ( v23 != -4096 && v23 != 0 )
      sub_BD60C0(v22);
    v3 = *(unsigned int *)(a1 + 928);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 912), 32 * v3, 8);
  v4 = *(_QWORD *)(a1 + 824);
  if ( v4 != a1 + 840 )
    _libc_free(v4);
  v5 = *(_QWORD *)(a1 + 744);
  if ( v5 != a1 + 760 )
    _libc_free(v5);
  if ( (*(_BYTE *)(a1 + 672) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 680), 16LL * *(unsigned int *)(a1 + 688), 8);
  v6 = *(__int64 **)(a1 + 576);
  v7 = &v6[*(unsigned int *)(a1 + 584)];
  if ( v6 != v7 )
  {
    for ( i = *(_QWORD *)(a1 + 576); ; i = *(_QWORD *)(a1 + 576) )
    {
      v9 = *v6;
      v10 = (unsigned int)(((__int64)v6 - i) >> 3) >> 7;
      v11 = 4096LL << v10;
      if ( v10 >= 0x1E )
        v11 = 0x40000000000LL;
      ++v6;
      sub_C7D6A0(v9, v11, 16);
      if ( v7 == v6 )
        break;
    }
  }
  v12 = *(__int64 **)(a1 + 624);
  v13 = (unsigned __int64)&v12[2 * *(unsigned int *)(a1 + 632)];
  if ( v12 != (__int64 *)v13 )
  {
    do
    {
      v14 = v12[1];
      v15 = *v12;
      v12 += 2;
      sub_C7D6A0(v15, v14, 16);
    }
    while ( (__int64 *)v13 != v12 );
    v13 = *(_QWORD *)(a1 + 624);
  }
  if ( v13 != a1 + 640 )
    _libc_free(v13);
  v16 = *(_QWORD *)(a1 + 576);
  if ( v16 != a1 + 592 )
    _libc_free(v16);
  sub_C7D6A0(*(_QWORD *)(a1 + 536), 40LL * *(unsigned int *)(a1 + 552), 8);
  sub_278E4C0(a1 + 312);
  v17 = *(_QWORD *)(a1 + 256);
  if ( v17 != a1 + 272 )
    _libc_free(v17);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 8LL * *(unsigned int *)(a1 + 248), 8);
  *(_QWORD *)a1 = &unk_49DAF80;
  return sub_BB9100(a1);
}
