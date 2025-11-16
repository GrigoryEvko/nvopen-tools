// Function: sub_CF5CA0
// Address: 0xcf5ca0
//
__int64 __fastcall sub_CF5CA0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // rax
  unsigned int v4; // r12d
  _QWORD v6[2]; // [rsp+0h] [rbp-2C0h] BYREF
  __int64 v7; // [rsp+10h] [rbp-2B0h]
  __int64 v8; // [rsp+18h] [rbp-2A8h] BYREF
  unsigned int v9; // [rsp+20h] [rbp-2A0h]
  _QWORD v10[2]; // [rsp+158h] [rbp-168h] BYREF
  char v11; // [rsp+168h] [rbp-158h]
  _BYTE *v12; // [rsp+170h] [rbp-150h]
  __int64 v13; // [rsp+178h] [rbp-148h]
  _BYTE v14[128]; // [rsp+180h] [rbp-140h] BYREF
  __int16 v15; // [rsp+200h] [rbp-C0h]
  _QWORD v16[2]; // [rsp+208h] [rbp-B8h] BYREF
  __int64 v17; // [rsp+218h] [rbp-A8h]
  __int64 v18; // [rsp+220h] [rbp-A0h] BYREF
  unsigned int v19; // [rsp+228h] [rbp-98h]
  char v20; // [rsp+2A0h] [rbp-20h] BYREF

  v2 = &v8;
  v6[0] = a1;
  v6[1] = 0;
  v7 = 1;
  do
  {
    *v2 = -4;
    v2 += 5;
    *(v2 - 4) = -3;
    *(v2 - 3) = -4;
    *(v2 - 2) = -3;
  }
  while ( v2 != v10 );
  v13 = 0x400000000LL;
  v10[0] = v16;
  v10[1] = 0;
  v11 = 0;
  v12 = v14;
  v16[1] = 0;
  v17 = 1;
  v15 = 256;
  v16[0] = &unk_49DDBE8;
  v3 = &v18;
  do
  {
    *v3 = -4096;
    v3 += 2;
  }
  while ( v3 != (__int64 *)&v20 );
  v4 = sub_CF5230(a1, a2, (__int64)v6);
  v16[0] = &unk_49DDBE8;
  if ( (v17 & 1) == 0 )
  {
    a2 = 16LL * v19;
    sub_C7D6A0(v18, a2, 8);
  }
  nullsub_184(v16);
  if ( v12 != v14 )
    _libc_free(v12, a2);
  if ( (v7 & 1) == 0 )
    sub_C7D6A0(v8, 40LL * v9, 8);
  return v4;
}
