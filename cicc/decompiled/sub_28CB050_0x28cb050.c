// Function: sub_28CB050
// Address: 0x28cb050
//
__int64 __fastcall sub_28CB050(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 *v3; // rax
  __int64 *v4; // rax
  __int64 v5; // r13
  _QWORD v7[3]; // [rsp+0h] [rbp-2D0h] BYREF
  __int64 v8; // [rsp+18h] [rbp-2B8h]
  __int64 v9; // [rsp+20h] [rbp-2B0h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-2A8h]
  _QWORD v11[2]; // [rsp+160h] [rbp-170h] BYREF
  char v12; // [rsp+170h] [rbp-160h]
  _BYTE *v13; // [rsp+178h] [rbp-158h]
  __int64 v14; // [rsp+180h] [rbp-150h]
  _BYTE v15[128]; // [rsp+188h] [rbp-148h] BYREF
  __int16 v16; // [rsp+208h] [rbp-C8h]
  _QWORD v17[2]; // [rsp+210h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+220h] [rbp-B0h]
  __int64 v19; // [rsp+228h] [rbp-A8h] BYREF
  unsigned int v20; // [rsp+230h] [rbp-A0h]
  char v21; // [rsp+2A8h] [rbp-28h] BYREF

  v2 = *a1[1];
  v7[2] = 0;
  v8 = 1;
  v7[0] = v2;
  v7[1] = v2;
  v3 = &v9;
  do
  {
    *v3 = -4;
    v3 += 5;
    *(v3 - 4) = -3;
    *(v3 - 3) = -4;
    *(v3 - 2) = -3;
  }
  while ( v3 != v11 );
  v14 = 0x400000000LL;
  v11[0] = v17;
  v11[1] = 0;
  v12 = 0;
  v13 = v15;
  v17[1] = 0;
  v18 = 1;
  v16 = 256;
  v17[0] = &unk_49DDBE8;
  v4 = &v19;
  do
  {
    *v4 = -4096;
    v4 += 2;
  }
  while ( v4 != (__int64 *)&v21 );
  v5 = ((__int64 (__fastcall *)(_QWORD **, __int64, _QWORD *))(*a1)[2])(a1, a2, v7);
  v17[0] = &unk_49DDBE8;
  if ( (v18 & 1) == 0 )
    sub_C7D6A0(v19, 16LL * v20, 8);
  nullsub_184();
  if ( v13 != v15 )
    _libc_free((unsigned __int64)v13);
  if ( (v8 & 1) == 0 )
    sub_C7D6A0(v9, 40LL * v10, 8);
  return v5;
}
