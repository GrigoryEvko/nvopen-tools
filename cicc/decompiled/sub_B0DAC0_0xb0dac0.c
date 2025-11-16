// Function: sub_B0DAC0
// Address: 0xb0dac0
//
__int64 __fastcall sub_B0DAC0(_QWORD *a1, char a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // rdx
  _QWORD *v7; // [rsp+0h] [rbp-70h] BYREF
  __int64 v8; // [rsp+8h] [rbp-68h]
  _QWORD v9[12]; // [rsp+10h] [rbp-60h] BYREF

  v7 = v9;
  v8 = 0x800000000LL;
  if ( (a2 & 1) != 0 )
  {
    v9[0] = 6;
    LODWORD(v8) = 1;
  }
  sub_AF6280((__int64)&v7, a3);
  if ( (a2 & 2) != 0 )
  {
    v5 = (unsigned int)v8;
    v6 = (unsigned int)v8 + 1LL;
    if ( v6 > HIDWORD(v8) )
    {
      sub_C8D5F0(&v7, v9, v6, 8);
      v5 = (unsigned int)v8;
    }
    v7[v5] = 6;
    LODWORD(v8) = v8 + 1;
  }
  v3 = sub_B0D8A0(a1, (__int64)&v7, (a2 & 4) != 0, (a2 & 8) != 0);
  if ( v7 != v9 )
    _libc_free(v7, &v7);
  return v3;
}
