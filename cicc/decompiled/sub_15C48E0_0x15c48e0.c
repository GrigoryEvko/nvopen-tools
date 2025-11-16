// Function: sub_15C48E0
// Address: 0x15c48e0
//
__int64 __fastcall sub_15C48E0(_QWORD *a1, char a2, __int64 a3, char a4, char a5)
{
  __int64 v7; // r12
  __int64 v9; // rax
  _QWORD *v10; // [rsp+0h] [rbp-80h] BYREF
  __int64 v11; // [rsp+8h] [rbp-78h]
  _QWORD v12[14]; // [rsp+10h] [rbp-70h] BYREF

  v10 = v12;
  v11 = 0x800000000LL;
  if ( a2 )
  {
    v12[0] = 6;
    LODWORD(v11) = 1;
  }
  sub_15B13F0((__int64)&v10, a3);
  if ( a4 )
  {
    v9 = (unsigned int)v11;
    if ( (unsigned int)v11 >= HIDWORD(v11) )
    {
      sub_16CD150(&v10, v12, 0, 8);
      v9 = (unsigned int)v11;
    }
    v10[v9] = 6;
    LODWORD(v11) = v11 + 1;
  }
  v7 = sub_15C46E0(a1, (__int64)&v10, a5);
  if ( v10 != v12 )
    _libc_free((unsigned __int64)v10);
  return v7;
}
