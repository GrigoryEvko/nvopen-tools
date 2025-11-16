// Function: sub_2E8AD10
// Address: 0x2e8ad10
//
__int64 __fastcall sub_2E8AD10(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 *v4; // rax
  __int64 *v5; // rax
  unsigned int v6; // r13d
  __int64 v8[3]; // [rsp-2D8h] [rbp-2D8h] BYREF
  __int64 v9; // [rsp-2C0h] [rbp-2C0h]
  __int64 v10; // [rsp-2B8h] [rbp-2B8h] BYREF
  unsigned int v11; // [rsp-2B0h] [rbp-2B0h]
  _QWORD v12[2]; // [rsp-178h] [rbp-178h] BYREF
  char v13; // [rsp-168h] [rbp-168h]
  _WORD *v14; // [rsp-160h] [rbp-160h]
  __int64 v15; // [rsp-158h] [rbp-158h]
  _WORD v16[68]; // [rsp-150h] [rbp-150h] BYREF
  _QWORD v17[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v18; // [rsp-B8h] [rbp-B8h]
  __int64 v19; // [rsp-B0h] [rbp-B0h] BYREF
  unsigned int v20; // [rsp-A8h] [rbp-A8h]
  __int64 v21; // [rsp-30h] [rbp-30h] BYREF

  if ( !a2 )
    return sub_2E8A880(a1, 0, a3, a4);
  v4 = &v10;
  v8[0] = a2;
  v8[1] = a2;
  v8[2] = 0;
  v9 = 1;
  do
  {
    *v4 = -4;
    v4 += 5;
    *(v4 - 4) = -3;
    *(v4 - 3) = -4;
    *(v4 - 2) = -3;
  }
  while ( v4 != v12 );
  v15 = 0x400000000LL;
  v12[0] = v17;
  v12[1] = 0;
  v13 = 0;
  v14 = v16;
  v17[1] = 0;
  v18 = 1;
  v16[64] = 256;
  v17[0] = &unk_49DDBE8;
  v5 = &v19;
  do
  {
    *v5 = -4096;
    v5 += 2;
  }
  while ( v5 != &v21 );
  v6 = sub_2E8A880(a1, v8, a3, a4);
  v17[0] = &unk_49DDBE8;
  if ( (v18 & 1) == 0 )
    sub_C7D6A0(v19, 16LL * v20, 8);
  nullsub_184();
  if ( v14 != v16 )
    _libc_free((unsigned __int64)v14);
  if ( (v9 & 1) == 0 )
    sub_C7D6A0(v10, 40LL * v11, 8);
  return v6;
}
