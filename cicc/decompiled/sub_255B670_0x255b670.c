// Function: sub_255B670
// Address: 0x255b670
//
_BOOL8 __fastcall sub_255B670(__int64 a1, __int64 a2)
{
  const void **v2; // r15
  _BOOL4 v3; // r14d
  void *v5; // [rsp+0h] [rbp-A0h] BYREF
  unsigned int v6; // [rsp+8h] [rbp-98h]
  const void *v7; // [rsp+10h] [rbp-90h] BYREF
  unsigned int v8; // [rsp+18h] [rbp-88h]
  void *v9; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v10; // [rsp+28h] [rbp-78h]
  const void *v11; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+38h] [rbp-68h]
  __int64 v13; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+48h] [rbp-58h]
  __int64 v15; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v16; // [rsp+58h] [rbp-48h]
  __int64 v17; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v18; // [rsp+68h] [rbp-38h]

  v2 = (const void **)(a1 + 16);
  v6 = *(_DWORD *)(a1 + 24);
  if ( v6 > 0x40 )
    sub_C43780((__int64)&v5, v2);
  else
    v5 = *(void **)(a1 + 16);
  v8 = *(_DWORD *)(a1 + 40);
  if ( v8 > 0x40 )
    sub_C43780((__int64)&v7, (const void **)(a1 + 32));
  else
    v7 = *(const void **)(a1 + 32);
  v10 = *(_DWORD *)(a2 + 24);
  if ( v10 > 0x40 )
    sub_C43780((__int64)&v9, (const void **)(a2 + 16));
  else
    v9 = *(void **)(a2 + 16);
  v12 = *(_DWORD *)(a2 + 40);
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)(a2 + 32));
  else
    v11 = *(const void **)(a2 + 32);
  sub_254F7F0(a1, (__int64)&v9);
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0((unsigned __int64)v11);
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0((unsigned __int64)v9);
  v9 = &unk_4A16D38;
  v10 = *(_DWORD *)(a1 + 8);
  v12 = *(_DWORD *)(a1 + 24);
  if ( v12 > 0x40 )
  {
    sub_C43780((__int64)&v11, v2);
    v14 = *(_DWORD *)(a1 + 40);
    if ( v14 <= 0x40 )
    {
LABEL_17:
      v13 = *(_QWORD *)(a1 + 32);
      v16 = *(_DWORD *)(a1 + 56);
      if ( v16 <= 0x40 )
        goto LABEL_18;
LABEL_38:
      sub_C43780((__int64)&v15, (const void **)(a1 + 48));
      v18 = *(_DWORD *)(a1 + 72);
      if ( v18 <= 0x40 )
        goto LABEL_19;
      goto LABEL_39;
    }
  }
  else
  {
    v11 = *(const void **)(a1 + 16);
    v14 = *(_DWORD *)(a1 + 40);
    if ( v14 <= 0x40 )
      goto LABEL_17;
  }
  sub_C43780((__int64)&v13, (const void **)(a1 + 32));
  v16 = *(_DWORD *)(a1 + 56);
  if ( v16 > 0x40 )
    goto LABEL_38;
LABEL_18:
  v15 = *(_QWORD *)(a1 + 48);
  v18 = *(_DWORD *)(a1 + 72);
  if ( v18 <= 0x40 )
  {
LABEL_19:
    v17 = *(_QWORD *)(a1 + 64);
    goto LABEL_20;
  }
LABEL_39:
  sub_C43780((__int64)&v17, (const void **)(a1 + 64));
LABEL_20:
  sub_253FFA0((__int64)&v9);
  v10 = *(_DWORD *)(a1 + 24);
  if ( v10 > 0x40 )
    sub_C43780((__int64)&v9, v2);
  else
    v9 = *(void **)(a1 + 16);
  v12 = *(_DWORD *)(a1 + 40);
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)(a1 + 32));
  else
    v11 = *(const void **)(a1 + 32);
  if ( v6 <= 0x40 )
  {
    v3 = 0;
    if ( v5 != v9 )
      goto LABEL_26;
  }
  else
  {
    v3 = 0;
    if ( !sub_C43C50((__int64)&v5, (const void **)&v9) )
      goto LABEL_26;
  }
  if ( v8 <= 0x40 )
    v3 = v7 == v11;
  else
    v3 = sub_C43C50((__int64)&v7, &v11);
LABEL_26:
  sub_969240((__int64 *)&v11);
  sub_969240((__int64 *)&v9);
  if ( v8 > 0x40 && v7 )
    j_j___libc_free_0_0((unsigned __int64)v7);
  if ( v6 > 0x40 && v5 )
    j_j___libc_free_0_0((unsigned __int64)v5);
  return v3;
}
