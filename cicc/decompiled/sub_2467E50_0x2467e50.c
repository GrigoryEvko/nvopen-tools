// Function: sub_2467E50
// Address: 0x2467e50
//
void __fastcall sub_2467E50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r12
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int *v16[2]; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v17; // [rsp-B8h] [rbp-B8h] BYREF
  _QWORD *v18; // [rsp-80h] [rbp-80h]
  void *v19; // [rsp-48h] [rbp-48h]

  if ( ((*(_WORD *)(*(_QWORD *)(a1 + 8) + 2LL) >> 4) & 0x3FF) != 0x4F )
  {
    v6 = *(unsigned int *)(a1 + 40);
    if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
    {
      sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v6 + 1, 8u, a5, a6);
      v6 = *(unsigned int *)(a1 + 40);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v6) = a2;
    ++*(_DWORD *)(a1 + 40);
    sub_23D0AB0((__int64)v16, a2, 0, 0, 0);
    v7 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v8 = *(_QWORD *)(a1 + 24);
    v9 = sub_BCB2B0(v18);
    if ( **(_BYTE **)(v8 + 8) )
      v10 = (__int64)sub_2465B30((__int64 *)v8, v7, (__int64)v16, v9, 1);
    else
      v10 = sub_2463FC0(v8, v7, v16, 0x103u);
    v11 = *(unsigned int *)(a1 + 176);
    v12 = sub_BCB2B0(v18);
    v13 = sub_AD6530(v12, v7);
    v14 = sub_BCB2E0(v18);
    v15 = sub_ACD640(v14, v11, 0);
    sub_B34240((__int64)v16, v10, v13, v15, 0x103u, 0, 0, 0, 0);
    nullsub_61();
    v19 = &unk_49DA100;
    nullsub_63();
    if ( (__int64 *)v16[0] != &v17 )
      _libc_free((unsigned __int64)v16[0]);
  }
}
