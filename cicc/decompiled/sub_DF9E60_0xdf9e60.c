// Function: sub_DF9E60
// Address: 0xdf9e60
//
__int64 __fastcall sub_DF9E60(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 (__fastcall *v12)(__int64, __int64, __int64, __int64, __int64, __int64, int, __int64); // r15
  void (__fastcall *v13)(_BYTE *, __int64, __int64); // rax
  __int64 v15; // rdx
  const void *v18; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v19; // [rsp+38h] [rbp-88h]
  __int64 v20; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v21; // [rsp+48h] [rbp-78h]
  _BYTE v22[16]; // [rsp+50h] [rbp-70h] BYREF
  void (__fastcall *v23)(__int64 *, __int64 *, __int64); // [rsp+60h] [rbp-60h]
  __int64 v24; // [rsp+68h] [rbp-58h]
  __int64 v25; // [rsp+70h] [rbp-50h] BYREF
  __int64 v26; // [rsp+78h] [rbp-48h]
  void (__fastcall *v27)(__int64 *, __int64 *, __int64); // [rsp+80h] [rbp-40h]
  __int64 v28; // [rsp+88h] [rbp-38h]

  v10 = *a1;
  v11 = **a1;
  v23 = 0;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, __int64, int, __int64))(v11 + 424);
  v13 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a8 + 16);
  if ( v13 )
  {
    v13(v22, a8, 2);
    v24 = *(_QWORD *)(a8 + 24);
    v23 = *(void (__fastcall **)(__int64 *, __int64 *, __int64))(a8 + 16);
  }
  v19 = *(_DWORD *)(a4 + 8);
  if ( v19 > 0x40 )
  {
    sub_C43780((__int64)&v18, (const void **)a4);
    if ( v12 == sub_DF6AF0 )
      goto LABEL_5;
LABEL_21:
    v25 = v12((__int64)v10, a2, a3, (__int64)&v18, a5, a6, a7, (__int64)v22);
    v26 = v15;
    goto LABEL_11;
  }
  v18 = *(const void **)a4;
  if ( v12 != sub_DF6AF0 )
    goto LABEL_21;
LABEL_5:
  v27 = 0;
  if ( v23 )
  {
    v23(&v25, (__int64 *)v22, 2);
    v28 = v24;
    v27 = v23;
  }
  v21 = v19;
  if ( v19 > 0x40 )
  {
    sub_C43780((__int64)&v20, &v18);
    if ( v21 > 0x40 )
    {
      if ( v20 )
        j_j___libc_free_0_0(v20);
    }
  }
  if ( v27 )
    v27(&v25, &v25, 3);
  LOBYTE(v26) = 0;
LABEL_11:
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v23 )
    v23((__int64 *)v22, (__int64 *)v22, 3);
  return v25;
}
