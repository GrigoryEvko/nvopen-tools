// Function: sub_CF5B00
// Address: 0xcf5b00
//
__int64 __fastcall sub_CF5B00(_QWORD *a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 *v3; // rax
  __int64 *v4; // rax
  unsigned int v5; // r12d
  _QWORD v7[2]; // [rsp+0h] [rbp-2C0h] BYREF
  __int64 v8; // [rsp+10h] [rbp-2B0h]
  __int64 v9; // [rsp+18h] [rbp-2A8h] BYREF
  unsigned int v10; // [rsp+20h] [rbp-2A0h]
  _QWORD v11[2]; // [rsp+158h] [rbp-168h] BYREF
  char v12; // [rsp+168h] [rbp-158h]
  _BYTE *v13; // [rsp+170h] [rbp-150h]
  __int64 v14; // [rsp+178h] [rbp-148h]
  _BYTE v15[128]; // [rsp+180h] [rbp-140h] BYREF
  __int16 v16; // [rsp+200h] [rbp-C0h]
  _QWORD v17[2]; // [rsp+208h] [rbp-B8h] BYREF
  __int64 v18; // [rsp+218h] [rbp-A8h]
  __int64 v19; // [rsp+220h] [rbp-A0h] BYREF
  unsigned int v20; // [rsp+228h] [rbp-98h]
  char v21; // [rsp+2A0h] [rbp-20h] BYREF

  v3 = &v9;
  v7[0] = a1;
  v7[1] = 0;
  v8 = 1;
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
  v5 = sub_CF5A30(a1, a2, a3, (__int64)v7);
  v17[0] = &unk_49DDBE8;
  if ( (v18 & 1) == 0 )
  {
    a2 = (unsigned __int8 *)(16LL * v20);
    sub_C7D6A0(v19, (__int64)a2, 8);
  }
  nullsub_184(v17);
  if ( v13 != v15 )
    _libc_free(v13, a2);
  if ( (v8 & 1) == 0 )
    sub_C7D6A0(v9, 40LL * v10, 8);
  return v5;
}
