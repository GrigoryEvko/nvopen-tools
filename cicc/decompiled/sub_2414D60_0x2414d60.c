// Function: sub_2414D60
// Address: 0x2414d60
//
void __fastcall sub_2414D60(__int64 **a1, char a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rdx
  __int64 v10; // rax
  int v11; // r14d
  unsigned __int64 *v12; // rdx
  char v13; // cl
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  __int64 *v16; // r13
  __int64 v17; // rdi
  __int64 **v18; // rax
  __int64 v19; // r14
  __int64 *v20; // r12
  __int64 v21; // r13
  __int64 v22; // [rsp+8h] [rbp-C8h] BYREF
  unsigned __int64 v23[2]; // [rsp+10h] [rbp-C0h] BYREF
  char v24; // [rsp+20h] [rbp-B0h] BYREF
  void *v25; // [rsp+90h] [rbp-40h]

  if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a3 - 8);
  else
    v5 = a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF);
  v6 = *(_QWORD *)(v5 + 32);
  v7 = sub_B43CC0(a3);
  v8 = sub_9208B0(v7, *(_QWORD *)(v6 + 8));
  v23[1] = v9;
  v23[0] = (unsigned __int64)(v8 + 7) >> 3;
  v10 = sub_CA1930(v23);
  v11 = v10;
  if ( v10 )
  {
    sub_23D0AB0((__int64)v23, a3, 0, 0, 0);
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
      v12 = *(unsigned __int64 **)(a3 - 8);
    else
      v12 = (unsigned __int64 *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
    v13 = 0;
    if ( (_BYTE)qword_4FE3A68 )
    {
      v13 = -1;
      if ( 1LL << a2 )
      {
        _BitScanReverse64(&v14, 1LL << a2);
        v13 = 63 - (v14 ^ 0x3F);
      }
    }
    sub_2413070(*a1, *v12, v11, v13, a3 + 24, 0);
    v15 = *(_QWORD *)(a3 + 8);
    v16 = *a1;
    v17 = **a1;
    if ( (unsigned __int8)(*(_BYTE *)(v15 + 8) - 15) > 1u )
    {
      v19 = *(_QWORD *)(v17 + 72);
    }
    else
    {
      v18 = (__int64 **)sub_240F000(v17, v15);
      v19 = sub_AC9350(v18);
    }
    v22 = a3;
    *sub_FAA780((__int64)(v16 + 22), &v22) = v19;
    v20 = *a1;
    v21 = *(_QWORD *)(*v20 + 40);
    if ( (unsigned __int8)sub_240D530() )
    {
      v22 = a3;
      *sub_FAA780((__int64)(v20 + 26), &v22) = v21;
    }
    nullsub_61();
    v25 = &unk_49DA100;
    nullsub_63();
    if ( (char *)v23[0] != &v24 )
      _libc_free(v23[0]);
  }
}
