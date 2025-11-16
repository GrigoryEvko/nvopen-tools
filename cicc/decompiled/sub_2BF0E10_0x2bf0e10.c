// Function: sub_2BF0E10
// Address: 0x2bf0e10
//
void __fastcall sub_2BF0E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // r14
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // r8
  __int64 *v19; // r12
  __int64 *v20; // r15
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // rdx
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 *v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h]
  _BYTE v27[96]; // [rsp+20h] [rbp-60h] BYREF

  v6 = (const void *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = 0x200000000LL;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  v8 = a1 + 160;
  *(_QWORD *)(a1 + 88) = 0x200000000LL;
  *(_QWORD *)(a1 + 152) = 0x200000000LL;
  v9 = a1 + 184;
  v10 = a1 + 216;
  *(_QWORD *)(v10 - 48) = v9;
  *(_QWORD *)(v10 - 72) = v8;
  *(_QWORD *)(v10 - 200) = v6;
  *(_QWORD *)(v10 - 168) = 0;
  *(_QWORD *)(v10 - 160) = 0;
  *(_QWORD *)(v10 - 152) = 0;
  *(_DWORD *)(v10 - 144) = 0;
  *(_QWORD *)(v10 - 104) = 0;
  *(_QWORD *)(v10 - 96) = 0;
  *(_QWORD *)(v10 - 88) = 0;
  *(_DWORD *)(v10 - 80) = 0;
  *(_QWORD *)(v10 - 40) = 0;
  *(_BYTE *)(v10 - 32) = 0;
  *(_QWORD *)(v10 - 16) = 0;
  *(_QWORD *)(v10 - 8) = 0;
  sub_2BF0340(v10, 0, 0, 0, a5, a6);
  sub_2BF0340(a1 + 272, 0, 0, 0, v11, v12);
  sub_2BF0340(a1 + 328, 0, 0, 0, v13, v14);
  *(_QWORD *)(a1 + 600) = 0x600000000LL;
  *(_QWORD *)(a1 + 416) = a1 + 432;
  *(_QWORD *)(a1 + 424) = 0x1000000000LL;
  *(_QWORD *)(a1 + 384) = 0;
  *(_QWORD *)(a1 + 392) = 0;
  *(_QWORD *)(a1 + 400) = 0;
  *(_DWORD *)(a1 + 408) = 0;
  *(_QWORD *)(a1 + 560) = 0;
  *(_QWORD *)(a1 + 568) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_DWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = a1 + 608;
  v15 = sub_D4B130(a2);
  v16 = sub_2BF0CC0(a1, v15);
  *(_QWORD *)a1 = v16;
  sub_2BF04F0(v16, a1);
  v17 = sub_2BF0CC0(a1, **(_QWORD **)(a2 + 32));
  v25 = (__int64 *)v27;
  *(_QWORD *)(a1 + 8) = v17;
  v26 = 0x600000000LL;
  sub_D472F0(a2, (__int64)&v25);
  v18 = v25;
  v19 = &v25[(unsigned int)v26];
  if ( v19 != v25 )
  {
    v20 = v25;
    do
    {
      v21 = sub_2BF0CC0(a1, *v20);
      v23 = *(unsigned int *)(a1 + 24);
      if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
      {
        v24 = v21;
        sub_C8D5F0(a1 + 16, v6, v23 + 1, 8u, v22, v23 + 1);
        v23 = *(unsigned int *)(a1 + 24);
        v21 = v24;
      }
      ++v20;
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8 * v23) = v21;
      ++*(_DWORD *)(a1 + 24);
    }
    while ( v19 != v20 );
    v18 = v25;
  }
  if ( v18 != (__int64 *)v27 )
    _libc_free((unsigned __int64)v18);
}
