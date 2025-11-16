// Function: sub_29AA880
// Address: 0x29aa880
//
__int64 *__fastcall sub_29AA880(_QWORD **a1, __int64 *a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  int v12; // edx
  unsigned __int64 v13; // rax
  __int64 *result; // rax
  __int64 v15; // [rsp+0h] [rbp-60h] BYREF
  _QWORD **v16; // [rsp+8h] [rbp-58h] BYREF
  __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-48h] BYREF
  _QWORD v19[8]; // [rsp+20h] [rbp-40h] BYREF

  v15 = a6;
  v16 = a1;
  v9 = sub_BCB2E0(*a1);
  v17 = sub_ACD640(v9, -1, 1u);
  v10 = *(_QWORD *)(v15 + 40);
  v11 = *(_QWORD *)(v10 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v11 == v10 + 48 )
  {
    v13 = 0;
  }
  else
  {
    if ( !v11 )
      BUG();
    v12 = *(unsigned __int8 *)(v11 - 24);
    v13 = v11 - 24;
    if ( (unsigned int)(v12 - 30) >= 0xB )
      v13 = 0;
  }
  v18 = v13;
  v19[0] = &v16;
  v19[1] = &v17;
  v19[2] = &v15;
  result = (__int64 *)&v18;
  v19[3] = &v18;
  if ( !a3 )
  {
    if ( !a5 )
      return result;
    return sub_29AA260((__int64)v19, 0xD2u, a4, a5, 0);
  }
  result = sub_29AA260((__int64)v19, 0xD3u, a2, a3, 1);
  if ( a5 )
    return sub_29AA260((__int64)v19, 0xD2u, a4, a5, 0);
  return result;
}
