// Function: sub_DF9D60
// Address: 0xdf9d60
//
__int64 __fastcall sub_DF9D60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  __int64 (__fastcall *v9)(__int64, __int64, __int64, __int64); // r15
  unsigned int v10; // eax
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-68h]
  const void *v14; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  __int64 v17; // [rsp+38h] [rbp-38h]

  v8 = *a1;
  v9 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)*a1 + 416LL);
  v15 = *(_DWORD *)(a4 + 8);
  if ( v15 > 0x40 )
  {
    v13 = a5;
    sub_C43780((__int64)&v14, (const void **)a4);
    a5 = v13;
    if ( v9 == sub_DF6950 )
      goto LABEL_3;
LABEL_14:
    v16 = ((__int64 (__fastcall *)(__int64, __int64, __int64, const void **, __int64, __int64))v9)(
            v8,
            a2,
            a3,
            &v14,
            a5,
            a6);
    v10 = v15;
    v17 = v12;
    goto LABEL_5;
  }
  v14 = *(const void **)a4;
  if ( v9 != sub_DF6950 )
    goto LABEL_14;
LABEL_3:
  v10 = v15;
  LODWORD(v17) = v15;
  if ( v15 > 0x40 )
  {
    sub_C43780((__int64)&v16, &v14);
    if ( (unsigned int)v17 > 0x40 && v16 )
      j_j___libc_free_0_0(v16);
    v10 = v15;
  }
  LOBYTE(v17) = 0;
LABEL_5:
  if ( v10 > 0x40 && v14 )
    j_j___libc_free_0_0(v14);
  return v16;
}
