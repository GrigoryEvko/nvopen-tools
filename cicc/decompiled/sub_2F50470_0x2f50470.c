// Function: sub_2F50470
// Address: 0x2f50470
//
__int64 sub_2F50470()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _BYTE v3[16]; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v4)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-20h]

  v4 = 0;
  v0 = sub_22077B0(0xE8u);
  v1 = v0;
  if ( v0 )
    sub_2F50330(v0, (__int64)v3);
  if ( v4 )
    v4(v3, v3, 3);
  return v1;
}
