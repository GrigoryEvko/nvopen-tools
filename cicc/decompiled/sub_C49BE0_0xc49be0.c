// Function: sub_C49BE0
// Address: 0xc49be0
//
__int64 __fastcall sub_C49BE0(__int64 a1, __int64 a2, __int64 a3, bool *a4)
{
  unsigned int v8; // ebx
  int v9; // edx
  unsigned __int64 v10; // rdx
  unsigned int v11; // eax
  unsigned int v12; // esi
  unsigned __int64 v13; // rax
  unsigned int v15; // esi
  __int64 v16; // rdx
  __int64 v17; // rax
  char v18; // di
  unsigned __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  int v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-48h]
  unsigned __int64 v25; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-38h]

  v8 = *(_DWORD *)(a2 + 8);
  if ( v8 <= 0x40 )
  {
    v9 = *(_DWORD *)(a2 + 8);
    if ( *(_QWORD *)a2 )
    {
      _BitScanReverse64(&v10, *(_QWORD *)a2);
      v9 = v8 - 64 + (v10 ^ 0x3F);
    }
    v11 = *(_DWORD *)(a3 + 8);
    if ( v11 <= 0x40 )
      goto LABEL_5;
LABEL_15:
    v22 = v9;
    v11 = sub_C444A0(a3);
    v9 = v22;
    goto LABEL_7;
  }
  v9 = sub_C444A0(a2);
  v11 = *(_DWORD *)(a3 + 8);
  if ( v11 > 0x40 )
    goto LABEL_15;
LABEL_5:
  v12 = v11 - 64;
  if ( *(_QWORD *)a3 )
  {
    _BitScanReverse64(&v13, *(_QWORD *)a3);
    v11 = v12 + (v13 ^ 0x3F);
  }
LABEL_7:
  if ( v11 + v9 + 2 <= v8 )
  {
    *a4 = 1;
    sub_C472A0(a1, a2, (__int64 *)a3);
    return a1;
  }
  v26 = v8;
  if ( v8 <= 0x40 )
  {
    v25 = *(_QWORD *)a2;
    goto LABEL_12;
  }
  sub_C43780((__int64)&v25, (const void **)a2);
  v8 = v26;
  if ( v26 <= 0x40 )
  {
LABEL_12:
    if ( v8 == 1 )
      v25 = 0;
    else
      v25 >>= 1;
    goto LABEL_18;
  }
  sub_C482E0((__int64)&v25, 1u);
LABEL_18:
  sub_C472A0((__int64)&v23, (__int64)&v25, (__int64 *)a3);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  v15 = v24;
  v16 = v23;
  v17 = 1LL << ((unsigned __int8)v24 - 1);
  v18 = (v24 - 1) & 0x3F;
  if ( v24 <= 0x40 )
  {
    *a4 = (v17 & v23) != 0;
    v20 = 0;
    if ( v15 != 1 )
    {
      v21 = (0xFFFFFFFFFFFFFFFFLL >> (63 - v18)) & (2 * v16);
      if ( v15 )
        v20 = v21;
    }
    v23 = v20;
  }
  else
  {
    *a4 = (*(_QWORD *)(v23 + 8LL * ((v24 - 1) >> 6)) & v17) != 0;
    sub_C47690(&v23, 1u);
  }
  v19 = *(_QWORD *)a2;
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
    v19 = *(_QWORD *)v19;
  if ( (v19 & 1) != 0 )
  {
    sub_C45EE0((__int64)&v23, (__int64 *)a3);
    if ( (int)sub_C49970((__int64)&v23, (unsigned __int64 *)a3) < 0 )
      *a4 = 1;
  }
  *(_DWORD *)(a1 + 8) = v24;
  *(_QWORD *)a1 = v23;
  return a1;
}
