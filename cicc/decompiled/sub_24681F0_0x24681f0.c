// Function: sub_24681F0
// Address: 0x24681f0
//
void __fastcall sub_24681F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 v4; // rcx
  __int64 v5; // r13
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int *v11[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v12; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v13; // [rsp-80h] [rbp-80h]
  void *v14; // [rsp-48h] [rbp-48h]

  if ( ((*(_WORD *)(*(_QWORD *)(a1 + 8) + 2LL) >> 4) & 0x3FF) != 0x4F )
  {
    sub_23D0AB0((__int64)v11, a2, 0, 0, 0);
    v2 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v3 = *(_QWORD *)(a1 + 24);
    v4 = sub_BCB2B0(v13);
    if ( **(_BYTE **)(v3 + 8) )
      v5 = (__int64)sub_2465B30((__int64 *)v3, v2, (__int64)v11, v4, 1);
    else
      v5 = sub_2463FC0(v3, v2, v11, 0x103u);
    v6 = *(unsigned int *)(a1 + 176);
    v7 = sub_BCB2B0(v13);
    v8 = sub_AD6530(v7, v2);
    v9 = sub_BCB2E0(v13);
    v10 = sub_ACD640(v9, v6, 0);
    sub_B34240((__int64)v11, v5, v8, v10, 0x103u, 0, 0, 0, 0);
    nullsub_61();
    v14 = &unk_49DA100;
    nullsub_63();
    if ( (__int64 *)v11[0] != &v12 )
      _libc_free((unsigned __int64)v11[0]);
  }
}
