// Function: sub_228F6B0
// Address: 0x228f6b0
//
__int64 __fastcall sub_228F6B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        char *a5,
        int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v10; // r12
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 *v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // r14d
  __int64 v18; // r9
  int v20; // eax
  int v21; // r14d
  __int64 v22; // r9
  unsigned int v23; // [rsp+Ch] [rbp-54h]
  __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 *v26; // [rsp+18h] [rbp-48h]
  __int64 *v27; // [rsp+18h] [rbp-48h]
  __int64 v28[8]; // [rsp+20h] [rbp-40h] BYREF

  v23 = a6 - 1;
  *(_BYTE *)(a7 + 43) = 0;
  v10 = sub_DCC810(*(__int64 **)(a1 + 8), a3, a4, 0, 0);
  v25 = *(_QWORD *)(a1 + 8);
  v11 = sub_D95540((__int64)v10);
  v12 = sub_DA2C50(v25, v11, 0, 0);
  sub_228CE50(a8, (__int64)v12, a2, (__int64)v10, (__int64)a5);
  if ( sub_228DFC0(a1, 0x20u, a3, a4) )
  {
    if ( *(_DWORD *)(a1 + 32) > v23 )
    {
      v18 = 16LL * v23;
      *(_BYTE *)(v18 + *(_QWORD *)(a7 + 48)) &= ~1u;
      *(_BYTE *)(*(_QWORD *)(a7 + 48) + v18) |= 0x10u;
    }
    return 0;
  }
  if ( *(_WORD *)(a2 + 24) )
    return 0;
  v26 = (__int64 *)a2;
  if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 8), a2) )
    v26 = sub_DCAF50(*(__int64 **)(a1 + 8), a2, 0);
  v13 = v10;
  if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 8), a2) )
    v13 = sub_DCAF50(*(__int64 **)(a1 + 8), (__int64)v10, 0);
  v14 = sub_D95540((__int64)v10);
  v15 = sub_228E360(a1, a5, v14);
  if ( !v15 )
  {
LABEL_14:
    if ( (unsigned __int8)sub_DBEC00(*(_QWORD *)(a1 + 8), (__int64)v13) )
      return 1;
    if ( !*((_WORD *)v10 + 12) )
    {
      sub_C4B8A0((__int64)v28, v10[4] + 24, *(_QWORD *)(a2 + 32) + 24LL);
      LOBYTE(v20) = sub_D94970((__int64)v28, 0);
      v21 = v20;
      sub_969240(v28);
      return v21 ^ 1u;
    }
    return 0;
  }
  v27 = sub_DCA690(*(__int64 **)(a1 + 8), (__int64)v26, (__int64)v15, 0, 0);
  LOBYTE(v16) = sub_228DFC0(a1, 0x26u, (__int64)v13, (__int64)v27);
  v17 = v16;
  if ( !(_BYTE)v16 )
  {
    if ( sub_228DFC0(a1, 0x20u, (__int64)v13, (__int64)v27) )
    {
      if ( *(_DWORD *)(a1 + 32) > v23 )
      {
        v22 = 16LL * v23;
        *(_BYTE *)(v22 + *(_QWORD *)(a7 + 48)) &= ~4u;
        *(_BYTE *)(*(_QWORD *)(a7 + 48) + v22) |= 0x20u;
        return v17;
      }
      return 0;
    }
    goto LABEL_14;
  }
  return 1;
}
