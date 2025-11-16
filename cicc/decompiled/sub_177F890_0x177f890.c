// Function: sub_177F890
// Address: 0x177f890
//
__int64 __fastcall sub_177F890(__int64 *a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  unsigned int *v5; // rax
  __int64 v6; // r12
  _BYTE v8[16]; // [rsp+0h] [rbp-30h] BYREF
  __int16 v9; // [rsp+10h] [rbp-20h]

  v5 = sub_177F600(*a1, a2, a3, a4);
  v9 = 257;
  v6 = sub_15FB440(24, a1, (__int64)v5, (__int64)v8, 0);
  if ( sub_15F23D0(a3) )
    sub_15F2350(v6, 1);
  return v6;
}
