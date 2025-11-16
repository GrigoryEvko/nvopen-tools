// Function: sub_259B800
// Address: 0x259b800
//
__int64 __fastcall sub_259B800(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  unsigned int v10; // r12d
  unsigned __int8 v12; // [rsp+Fh] [rbp-51h] BYREF
  __m128i v13; // [rsp+10h] [rbp-50h] BYREF
  __int64 v14; // [rsp+20h] [rbp-40h]
  unsigned __int64 v15[2]; // [rsp+28h] [rbp-38h] BYREF
  _BYTE v16[40]; // [rsp+38h] [rbp-28h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(unsigned int *)(a2 + 16);
  v15[0] = (unsigned __int64)v16;
  v15[1] = 0;
  v14 = v7;
  if ( (_DWORD)v8 )
  {
    sub_2538240((__int64)v15, (char **)(a2 + 8), a3, v8, a5, a6);
    v7 = v14;
  }
  v9 = sub_B43CB0(v7);
  sub_250D230((unsigned __int64 *)&v13, v9, 4, 0);
  v10 = sub_259B4A0(*a1, a1[1], &v13, 2, &v12, 0, 0);
  if ( (_BYTE)v10 )
    v10 = v12;
  if ( (_BYTE *)v15[0] != v16 )
    _libc_free(v15[0]);
  return v10;
}
