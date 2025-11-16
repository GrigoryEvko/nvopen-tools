// Function: sub_F0C5B0
// Address: 0xf0c5b0
//
__int64 __fastcall sub_F0C5B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v10; // rdi
  __int64 **v12; // r11
  void (__fastcall *v13)(_BYTE *, __int64, __int64); // rax
  __int64 v14; // rdx
  __int64 **v15; // [rsp+0h] [rbp-A0h]
  __int64 **v16; // [rsp+8h] [rbp-98h]
  const void *v19; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v20; // [rsp+38h] [rbp-68h]
  __int64 v21; // [rsp+40h] [rbp-60h]
  __int64 v22; // [rsp+48h] [rbp-58h]
  _BYTE v23[16]; // [rsp+50h] [rbp-50h] BYREF
  void (__fastcall *v24)(_BYTE *, _BYTE *, __int64); // [rsp+60h] [rbp-40h]
  __int64 v25; // [rsp+68h] [rbp-38h]

  v10 = *(_QWORD *)(a2 - 32);
  if ( v10 )
  {
    if ( *(_BYTE *)v10 )
    {
      v10 = 0;
    }
    else if ( *(_QWORD *)(v10 + 24) != *(_QWORD *)(a2 + 80) )
    {
      v10 = 0;
    }
  }
  if ( (unsigned __int8)sub_B2DD60(v10) )
  {
    v24 = 0;
    v12 = *(__int64 ***)(a1 + 8);
    v13 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(a7 + 16);
    if ( v13 )
    {
      v16 = *(__int64 ***)(a1 + 8);
      v13(v23, a7, 2);
      v12 = v16;
      v25 = *(_QWORD *)(a7 + 24);
      v24 = *(void (__fastcall **)(_BYTE *, _BYTE *, __int64))(a7 + 16);
    }
    v20 = *(_DWORD *)(a3 + 8);
    if ( v20 > 0x40 )
    {
      v15 = v12;
      sub_C43780((__int64)&v19, (const void **)a3);
      v12 = v15;
    }
    else
    {
      v19 = *(const void **)a3;
    }
    v21 = sub_DF9E60(v12, a1, a2, (__int64)&v19, a4, a5, a6, (__int64)v23);
    v22 = v14;
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    if ( v24 )
      v24(v23, v23, 3);
  }
  else
  {
    LOBYTE(v22) = 0;
  }
  return v21;
}
