// Function: sub_DF6AF0
// Address: 0xdf6af0
//
__int64 __fastcall sub_DF6AF0(
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
  __int64 v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v12; // [rsp+18h] [rbp-48h]
  __int64 v13; // [rsp+20h] [rbp-40h]
  __int64 v14; // [rsp+28h] [rbp-38h]
  _BYTE v15[16]; // [rsp+30h] [rbp-30h] BYREF
  void (__fastcall *v16)(_BYTE *, _BYTE *, __int64); // [rsp+40h] [rbp-20h]
  __int64 v17; // [rsp+48h] [rbp-18h]

  v16 = 0;
  v9 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a8 + 16);
  if ( v9 )
  {
    v9(v15, a8, 2);
    v17 = *(_QWORD *)(a8 + 24);
    v16 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a8 + 16);
  }
  v12 = *(_DWORD *)(a4 + 8);
  if ( v12 > 0x40 )
  {
    sub_C43780((__int64)&v11, (const void **)a4);
    LOBYTE(v14) = 0;
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
  }
  else
  {
    LOBYTE(v14) = 0;
  }
  if ( v16 )
    v16(v15, v15, 3);
  return v13;
}
