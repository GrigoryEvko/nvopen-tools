// Function: sub_103E0F0
// Address: 0x103e0f0
//
__int64 __fastcall sub_103E0F0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v6; // r13
  __int64 v7; // r14
  unsigned __int8 *v8; // rax
  size_t v9; // rdx
  void *v10; // rdi
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rsi
  size_t v16; // [rsp+0h] [rbp-300h]
  _QWORD v18[6]; // [rsp+10h] [rbp-2F0h] BYREF
  __int64 v19; // [rsp+40h] [rbp-2C0h]
  __int64 v20; // [rsp+48h] [rbp-2B8h] BYREF
  unsigned int v21; // [rsp+50h] [rbp-2B0h]
  _QWORD v22[2]; // [rsp+188h] [rbp-178h] BYREF
  char v23; // [rsp+198h] [rbp-168h]
  _BYTE *v24; // [rsp+1A0h] [rbp-160h]
  __int64 v25; // [rsp+1A8h] [rbp-158h]
  _BYTE v26[128]; // [rsp+1B0h] [rbp-150h] BYREF
  __int16 v27; // [rsp+230h] [rbp-D0h]
  _QWORD v28[2]; // [rsp+238h] [rbp-C8h] BYREF
  __int64 v29; // [rsp+248h] [rbp-B8h]
  __int64 v30; // [rsp+250h] [rbp-B0h] BYREF
  unsigned int v31; // [rsp+258h] [rbp-A8h]
  char v32; // [rsp+2D0h] [rbp-30h] BYREF

  v6 = *(__int64 **)(sub_BC1CD0(a4, &unk_4F8F810, a3) + 8);
  v7 = sub_904010(*a2, "MemorySSA (walker) for function: ");
  v8 = (unsigned __int8 *)sub_BD5D20(a3);
  v10 = *(void **)(v7 + 32);
  if ( *(_QWORD *)(v7 + 24) - (_QWORD)v10 < v9 )
  {
    v7 = sub_CB6200(v7, v8, v9);
  }
  else if ( v9 )
  {
    v16 = v9;
    memcpy(v10, v8, v9);
    *(_QWORD *)(v7 + 32) += v16;
  }
  sub_904010(v7, "\n");
  v18[0] = off_49E5A60;
  v18[1] = v6;
  v18[2] = sub_103E0E0(v6);
  v11 = *v6;
  v18[5] = 0;
  v19 = 1;
  v18[3] = v11;
  v18[4] = v11;
  v12 = &v20;
  do
  {
    *v12 = -4;
    v12 += 5;
    *(v12 - 4) = -3;
    *(v12 - 3) = -4;
    *(v12 - 2) = -3;
  }
  while ( v12 != v22 );
  v22[1] = 0;
  v24 = v26;
  v25 = 0x400000000LL;
  v27 = 256;
  v22[0] = v28;
  v23 = 0;
  v28[1] = 0;
  v29 = 1;
  v28[0] = &unk_49DDBE8;
  v13 = &v30;
  do
  {
    *v13 = -4096;
    v13 += 2;
  }
  while ( v13 != (__int64 *)&v32 );
  v14 = *a2;
  sub_A68C30(a3, *a2, (__int64)v18, 0, 0);
  *(_BYTE *)(a1 + 76) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 16) = 0x100000002LL;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 32) = &unk_4F82400;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)a1 = 1;
  v18[0] = off_49E5A60;
  v28[0] = &unk_49DDBE8;
  if ( (v29 & 1) == 0 )
  {
    v14 = 16LL * v31;
    sub_C7D6A0(v30, v14, 8);
  }
  nullsub_184();
  if ( v24 != v26 )
    _libc_free(v24, v14);
  if ( (v19 & 1) == 0 )
    sub_C7D6A0(v20, 40LL * v21, 8);
  nullsub_35();
  return a1;
}
