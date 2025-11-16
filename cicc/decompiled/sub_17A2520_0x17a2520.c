// Function: sub_17A2520
// Address: 0x17a2520
//
__int64 __fastcall sub_17A2520(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  _BYTE *v7; // r14
  unsigned __int8 v8; // al
  _BYTE *v9; // r15
  char v10; // r8
  __int64 result; // rax
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 *v17; // rbx
  __int64 v18; // rcx
  unsigned __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+18h] [rbp-48h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v25; // [rsp+28h] [rbp-38h]

  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v5 = *(_QWORD *)(a1 - 8);
  else
    v5 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v6 = 24LL * a2;
  v7 = *(_BYTE **)(v5 + v6);
  v8 = v7[16];
  v9 = v7 + 24;
  if ( v8 != 13 )
  {
    if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) != 16 )
      return 0;
    if ( v8 > 0x10u )
      return 0;
    v21 = sub_15A1020(v7, a2, *(_QWORD *)v7, a4);
    if ( !v21 )
      return 0;
    v9 = (_BYTE *)(v21 + 24);
    if ( *(_BYTE *)(v21 + 16) != 13 )
      return 0;
  }
  if ( *((_DWORD *)v9 + 2) <= 0x40u )
  {
    result = 0;
    if ( (*(_QWORD *)v9 & ~*a3) == 0 )
      return result;
  }
  else
  {
    v10 = sub_16A5A00((__int64 *)v9, a3);
    result = 0;
    if ( v10 )
      return result;
  }
  v12 = *((_DWORD *)v9 + 2);
  v23 = v12;
  if ( v12 > 0x40 )
  {
    sub_16A4FD0((__int64)&v22, (const void **)v9);
    v12 = v23;
    if ( v23 > 0x40 )
    {
      sub_16A8890(&v22, a3);
      v12 = v23;
      v14 = v22;
      goto LABEL_9;
    }
    v13 = v22;
  }
  else
  {
    v13 = *(_QWORD *)v9;
  }
  v14 = *a3 & v13;
  v22 = v14;
LABEL_9:
  v25 = v12;
  v24 = v14;
  v23 = 0;
  v15 = sub_15A1070(*(_QWORD *)v7, (__int64)&v24);
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v16 = *(_QWORD *)(a1 - 8);
  else
    v16 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
  v17 = (__int64 *)(v6 + v16);
  if ( *v17 )
  {
    v18 = v17[1];
    v19 = v17[2] & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v19 = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = *(_QWORD *)(v18 + 16) & 3LL | v19;
  }
  *v17 = v15;
  if ( v15 )
  {
    v20 = *(_QWORD *)(v15 + 8);
    v17[1] = v20;
    if ( v20 )
      *(_QWORD *)(v20 + 16) = (unsigned __int64)(v17 + 1) | *(_QWORD *)(v20 + 16) & 3LL;
    v17[2] = (v15 + 8) | v17[2] & 3;
    *(_QWORD *)(v15 + 8) = v17;
  }
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 )
  {
    if ( v22 )
      j_j___libc_free_0_0(v22);
  }
  return 1;
}
