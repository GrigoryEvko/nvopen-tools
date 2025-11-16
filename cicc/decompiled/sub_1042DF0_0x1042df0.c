// Function: sub_1042DF0
// Address: 0x1042df0
//
void __fastcall sub_1042DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 *v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 *v9; // rsi
  __int64 v10[3]; // [rsp+0h] [rbp-2E0h] BYREF
  __int64 v11; // [rsp+18h] [rbp-2C8h]
  __int64 v12; // [rsp+20h] [rbp-2C0h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-2B8h]
  _QWORD v14[2]; // [rsp+160h] [rbp-180h] BYREF
  char v15; // [rsp+170h] [rbp-170h]
  _BYTE *v16; // [rsp+178h] [rbp-168h]
  __int64 v17; // [rsp+180h] [rbp-160h]
  _BYTE v18[128]; // [rsp+188h] [rbp-158h] BYREF
  __int16 v19; // [rsp+208h] [rbp-D8h]
  _QWORD v20[2]; // [rsp+210h] [rbp-D0h] BYREF
  __int64 v21; // [rsp+220h] [rbp-C0h]
  __int64 v22; // [rsp+228h] [rbp-B8h] BYREF
  unsigned int v23; // [rsp+230h] [rbp-B0h]
  char v24; // [rsp+2A8h] [rbp-38h] BYREF

  *(_QWORD *)(a1 + 144) = a1 + 168;
  v5 = &v12;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = a2;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_DWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 16;
  *(_DWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 164) = 1;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_BYTE *)(a1 + 356) = 0;
  v10[2] = 0;
  v11 = 1;
  v10[0] = a3;
  v10[1] = a3;
  do
  {
    *v5 = -4;
    v5 += 5;
    *(v5 - 4) = -3;
    *(v5 - 3) = -4;
    *(v5 - 2) = -3;
  }
  while ( v5 != v14 );
  v17 = 0x400000000LL;
  v14[0] = v20;
  v14[1] = 0;
  v15 = 0;
  v16 = v18;
  v20[1] = 0;
  v21 = 1;
  v19 = 256;
  v20[0] = &unk_49DDBE8;
  v6 = &v22;
  do
  {
    *v6 = -4096;
    v6 += 2;
  }
  while ( v6 != (__int64 *)&v24 );
  v7 = *(_QWORD *)(a2 + 80);
  v8 = a2 + 72;
  v9 = v10;
  sub_1042620(a1, v10, v7, v8);
  *(_QWORD *)a1 = a3;
  sub_103E0E0((_QWORD *)a1);
  v20[0] = &unk_49DDBE8;
  if ( (v21 & 1) == 0 )
  {
    v9 = (__int64 *)(16LL * v23);
    sub_C7D6A0(v22, (__int64)v9, 8);
  }
  nullsub_184();
  if ( v16 != v18 )
    _libc_free(v16, v9);
  if ( (v11 & 1) == 0 )
    sub_C7D6A0(v12, 40LL * v13, 8);
}
