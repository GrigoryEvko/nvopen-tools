// Function: sub_277B150
// Address: 0x277b150
//
__int64 __fastcall sub_277B150(_QWORD *a1, __int64 a2)
{
  __int64 *v2; // rcx
  __int64 v3; // rax
  __int64 *v4; // rax
  __int64 *v5; // rax
  __int64 v6; // r8
  int v7; // ecx
  int v8; // ecx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // r14
  int v15; // eax
  int v16; // r10d
  _QWORD v17[3]; // [rsp+0h] [rbp-2D0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-2B8h]
  __int64 v19; // [rsp+20h] [rbp-2B0h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-2A8h]
  _QWORD v21[2]; // [rsp+160h] [rbp-170h] BYREF
  char v22; // [rsp+170h] [rbp-160h]
  _BYTE *v23; // [rsp+178h] [rbp-158h]
  __int64 v24; // [rsp+180h] [rbp-150h]
  _BYTE v25[128]; // [rsp+188h] [rbp-148h] BYREF
  __int16 v26; // [rsp+208h] [rbp-C8h]
  _QWORD v27[2]; // [rsp+210h] [rbp-C0h] BYREF
  __int64 v28; // [rsp+220h] [rbp-B0h]
  __int64 v29; // [rsp+228h] [rbp-A8h] BYREF
  unsigned int v30; // [rsp+230h] [rbp-A0h]
  char v31; // [rsp+2A8h] [rbp-28h] BYREF

  v2 = (__int64 *)a1[1];
  v3 = *v2;
  v17[2] = 0;
  v18 = 1;
  v17[0] = v3;
  v17[1] = v3;
  v4 = &v19;
  do
  {
    *v4 = -4;
    v4 += 5;
    *(v4 - 4) = -3;
    *(v4 - 3) = -4;
    *(v4 - 2) = -3;
  }
  while ( v4 != v21 );
  v24 = 0x400000000LL;
  v21[0] = v27;
  v21[1] = 0;
  v22 = 0;
  v23 = v25;
  v27[1] = 0;
  v28 = 1;
  v26 = 256;
  v27[0] = &unk_49DDBE8;
  v5 = &v29;
  do
  {
    *v5 = -4096;
    v5 += 2;
  }
  while ( v5 != (__int64 *)&v31 );
  v6 = v2[5];
  v7 = *((_DWORD *)v2 + 14);
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
    {
LABEL_7:
      v12 = v10[1];
      goto LABEL_8;
    }
    v15 = 1;
    while ( v11 != -4096 )
    {
      v16 = v15 + 1;
      v9 = v8 & (v15 + v9);
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_7;
      v15 = v16;
    }
  }
  v12 = 0;
LABEL_8:
  v13 = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD *))(*a1 + 16LL))(a1, v12, v17);
  v27[0] = &unk_49DDBE8;
  if ( (v28 & 1) == 0 )
    sub_C7D6A0(v29, 16LL * v30, 8);
  nullsub_184();
  if ( v23 != v25 )
    _libc_free((unsigned __int64)v23);
  if ( (v18 & 1) == 0 )
    sub_C7D6A0(v19, 40LL * v20, 8);
  return v13;
}
