// Function: sub_246D160
// Address: 0x246d160
//
__int64 __fastcall sub_246D160(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  unsigned int v6; // r12d
  __int64 *v8; // [rsp+0h] [rbp-80h]
  _BYTE *v9[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-60h] BYREF
  __int64 v11[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v12[8]; // [rsp+40h] [rbp-40h] BYREF

  v8 = sub_C60B10();
  v11[0] = (__int64)v12;
  sub_2462320(v11, a3, (__int64)&a3[a4]);
  v9[0] = v10;
  sub_2462320((__int64 *)v9, a1, (__int64)&a1[a2]);
  v6 = sub_CF9810((__int64)v8, v9, (__int64)v11);
  if ( (_QWORD *)v9[0] != v10 )
    j_j___libc_free_0((unsigned __int64)v9[0]);
  if ( (_QWORD *)v11[0] != v12 )
    j_j___libc_free_0(v11[0]);
  return v6;
}
