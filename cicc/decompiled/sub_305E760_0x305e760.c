// Function: sub_305E760
// Address: 0x305e760
//
__int64 __fastcall sub_305E760(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7,
        __int64 a8)
{
  void (__fastcall *v9)(_BYTE *, __int64, __int64); // rax
  const void *v11; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-68h]
  unsigned __int64 v13; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v14; // [rsp+28h] [rbp-58h]
  _BYTE v15[16]; // [rsp+30h] [rbp-50h] BYREF
  void (__fastcall *v16)(__int64 *, __int64 *, __int64); // [rsp+40h] [rbp-40h]
  __int64 v17; // [rsp+48h] [rbp-38h]
  __int64 v18; // [rsp+50h] [rbp-30h] BYREF
  __int64 v19; // [rsp+58h] [rbp-28h]
  void (__fastcall *v20)(__int64 *, __int64 *, __int64); // [rsp+60h] [rbp-20h]
  __int64 v21; // [rsp+68h] [rbp-18h]

  v16 = 0;
  v9 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a8 + 16);
  if ( v9 )
  {
    v9(v15, a8, 2);
    v17 = *(_QWORD *)(a8 + 24);
    v16 = *(void (__fastcall **)(__int64 *, __int64 *, __int64))(a8 + 16);
  }
  v12 = *(_DWORD *)(a4 + 8);
  if ( v12 > 0x40 )
    sub_C43780((__int64)&v11, (const void **)a4);
  else
    v11 = *(const void **)a4;
  v20 = 0;
  if ( v16 )
  {
    v16(&v18, (__int64 *)v15, 2);
    v21 = v17;
    v20 = v16;
  }
  v14 = v12;
  if ( v12 > 0x40 )
  {
    sub_C43780((__int64)&v13, &v11);
    if ( v14 > 0x40 )
    {
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
  }
  if ( v20 )
    v20(&v18, &v18, 3);
  LOBYTE(v19) = 0;
  if ( v12 > 0x40 && v11 )
    j_j___libc_free_0_0((unsigned __int64)v11);
  if ( v16 )
    v16((__int64 *)v15, (__int64 *)v15, 3);
  return v18;
}
