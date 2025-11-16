// Function: sub_1037870
// Address: 0x1037870
//
__int64 __fastcall sub_1037870(
        __int64 a1,
        __int64 a2,
        char a3,
        _QWORD *a4,
        __int16 a5,
        __int64 a6,
        _BYTE *a7,
        int *a8,
        char a9)
{
  __int64 v9; // rax
  __int64 *v10; // rax
  __int64 *v11; // rax
  __int64 v12; // r13
  __int64 v14[3]; // [rsp+0h] [rbp-2D0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-2B8h]
  __int64 v16; // [rsp+20h] [rbp-2B0h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-2A8h]
  _QWORD v18[2]; // [rsp+160h] [rbp-170h] BYREF
  char v19; // [rsp+170h] [rbp-160h]
  _BYTE *v20; // [rsp+178h] [rbp-158h]
  __int64 v21; // [rsp+180h] [rbp-150h]
  _BYTE v22[128]; // [rsp+188h] [rbp-148h] BYREF
  __int16 v23; // [rsp+208h] [rbp-C8h]
  void *v24; // [rsp+210h] [rbp-C0h]
  __int64 v25; // [rsp+218h] [rbp-B8h]
  __int64 v26; // [rsp+220h] [rbp-B0h]
  __int64 v27; // [rsp+228h] [rbp-A8h] BYREF
  unsigned int v28; // [rsp+230h] [rbp-A0h]
  char v29; // [rsp+2A8h] [rbp-28h] BYREF

  v9 = *(_QWORD *)(a1 + 256);
  v14[2] = 0;
  v15 = 1;
  v14[0] = v9;
  v14[1] = v9;
  v10 = &v16;
  do
  {
    *v10 = -4;
    v10 += 5;
    *(v10 - 4) = -3;
    *(v10 - 3) = -4;
    *(v10 - 2) = -3;
  }
  while ( v10 != v18 );
  v18[0] = a1 + 416;
  v21 = 0x400000000LL;
  v18[1] = 0;
  v19 = 0;
  v20 = v22;
  v25 = 0;
  v26 = 1;
  v23 = 256;
  v24 = &unk_49DDBE8;
  v11 = &v27;
  do
  {
    *v11 = -4096;
    v11 += 2;
  }
  while ( v11 != (__int64 *)&v29 );
  v12 = sub_1034B30(a1, a2, a3, a4, a5, a6, a7, a8, a9, v14);
  v24 = &unk_49DDBE8;
  if ( (v26 & 1) == 0 )
  {
    a2 = 16LL * v28;
    sub_C7D6A0(v27, a2, 8);
  }
  nullsub_184();
  if ( v20 != v22 )
    _libc_free(v20, a2);
  if ( (v15 & 1) == 0 )
    sub_C7D6A0(v16, 40LL * v17, 8);
  return v12;
}
