// Function: sub_2CD3E30
// Address: 0x2cd3e30
//
void __fastcall sub_2CD3E30(_BYTE *a1, _QWORD *a2, __int64 a3, unsigned __int64 a4)
{
  __int64 *v7; // rdi
  __int64 v8; // r10
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // r13
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // [rsp+0h] [rbp-110h]
  __int64 v15; // [rsp+18h] [rbp-F8h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-F0h] BYREF
  _QWORD v17[2]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v18; // [rsp+40h] [rbp-D0h]
  unsigned int *v19[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v20; // [rsp+60h] [rbp-B0h] BYREF
  void *v21; // [rsp+D0h] [rbp-40h]

  sub_23D0AB0((__int64)v19, (__int64)a2, 0, 0, 0);
  v7 = (__int64 *)a2[1];
  v8 = *(_QWORD *)(*(_QWORD *)(a2[5] + 72LL) + 40LL);
  v9 = *(_QWORD *)(*(a2 - 4) + 8LL);
  v15 = *(a2 - 4);
  v16[0] = v17;
  v17[0] = v9;
  v14 = v8;
  v16[1] = 0x100000001LL;
  v10 = sub_BCF480(v7, v17, 1, 0);
  v11 = sub_BA8C10(v14, a3, a4, v10, 0);
  v18 = 257;
  v13 = sub_921880(v19, v11, v12, (int)&v15, 1, (__int64)v16, 0);
  sub_BD84D0((__int64)a2, v13);
  sub_B43D60(a2);
  *a1 = 1;
  nullsub_61();
  v21 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v19[0] != &v20 )
    _libc_free((unsigned __int64)v19[0]);
}
