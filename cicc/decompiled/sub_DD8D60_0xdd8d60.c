// Function: sub_DD8D60
// Address: 0xdd8d60
//
__int64 *__fastcall sub_DD8D60(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, _BYTE *a5)
{
  __int64 *v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r14
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  __int64 *v18; // [rsp+8h] [rbp-58h]
  _QWORD v19[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v20[8]; // [rsp+20h] [rbp-40h] BYREF

  if ( !sub_BCAC40(*(_QWORD *)(a2 + 8), 1) || *a4 != 17 && *a5 != 17 )
    return sub_DA3860(a1, a2);
  v9 = sub_DD8400((__int64)a1, a3);
  v18 = sub_DD8400((__int64)a1, (__int64)a4);
  v10 = sub_DD8400((__int64)a1, (__int64)a5);
  v14 = (__int64)v18;
  v15 = (__int64)v10;
  if ( *((_WORD *)v18 + 12) )
  {
    if ( *((_WORD *)v10 + 12) )
      return sub_DA3860(a1, a2);
  }
  else
  {
    v9 = sub_DD1D00(a1, v9, v11, v12, v13);
    v14 = v15;
    v15 = (__int64)v18;
  }
  v16 = sub_DCC810(a1, v14, v15, 0, 0);
  v20[1] = sub_DCEE80(a1, (__int64)v9, (__int64)v16, 1);
  v19[0] = v20;
  v20[0] = v15;
  v19[1] = 0x200000002LL;
  v17 = sub_DC7EB0(a1, (__int64)v19, 0, 0);
  if ( (_QWORD *)v19[0] != v20 )
    _libc_free(v19[0], v19);
  return v17;
}
