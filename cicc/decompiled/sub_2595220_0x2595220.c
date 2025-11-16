// Function: sub_2595220
// Address: 0x2595220
//
__int64 __fastcall sub_2595220(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // r8
  __int64 v7; // rcx
  unsigned int v8; // r12d
  char v10; // [rsp+Fh] [rbp-51h] BYREF
  __m128i v11; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v12; // [rsp+20h] [rbp-40h]
  unsigned __int64 v13[2]; // [rsp+28h] [rbp-38h] BYREF
  _BYTE v14[40]; // [rsp+38h] [rbp-28h] BYREF

  v6 = *(_QWORD *)a2;
  v7 = *(unsigned int *)(a2 + 16);
  v13[0] = (unsigned __int64)v14;
  v13[1] = 0;
  v12 = v6;
  if ( (_DWORD)v7 )
  {
    sub_2538240((__int64)v13, (char **)(a2 + 8), a3, v7, v6, a6);
    v6 = v12;
  }
  sub_250D230((unsigned __int64 *)&v11, v6, 5, 0);
  v8 = sub_25950E0(*a1, a1[1], &v11, 0, &v10, 1, 0);
  if ( (_BYTE *)v13[0] != v14 )
    _libc_free(v13[0]);
  return v8;
}
