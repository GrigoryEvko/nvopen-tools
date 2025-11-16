// Function: sub_DCFA50
// Address: 0xdcfa50
//
_QWORD *__fastcall sub_DCFA50(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdi
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r14
  unsigned int v10; // r15d
  unsigned __int64 v11; // rax
  unsigned int v12; // r15d
  _QWORD *v13; // rax
  __int64 v14; // rax
  _QWORD *v15; // rax
  __int64 v17; // rax
  __int64 *v18; // r14
  __int64 v19; // rax
  _QWORD v20[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v21[8]; // [rsp+10h] [rbp-40h] BYREF

  if ( *(_WORD *)(a3 + 24) )
    goto LABEL_11;
  v5 = *(_QWORD *)(a3 + 32);
  v6 = *(_DWORD *)(v5 + 32);
  if ( v6 > 0x40 )
  {
    if ( (unsigned int)sub_C444A0(v5 + 24) != v6 - 1 )
    {
      if ( (unsigned int)sub_C44630(v5 + 24) == 1 )
        goto LABEL_5;
      goto LABEL_11;
    }
LABEL_14:
    v19 = sub_D95540(a2);
    return sub_DA2C50((__int64)a1, v19, 0, 0);
  }
  v17 = *(_QWORD *)(v5 + 24);
  if ( v17 == 1 )
    goto LABEL_14;
  if ( v17 && (v17 & (v17 - 1)) == 0 )
  {
LABEL_5:
    v7 = sub_D95540(a2);
    v8 = *(_QWORD *)(a3 + 32);
    v9 = v7;
    v10 = *(_DWORD *)(v8 + 32);
    if ( v10 > 0x40 )
    {
      v12 = v10 - 1 - sub_C444A0(v8 + 24);
    }
    else
    {
      v11 = *(_QWORD *)(v8 + 24);
      v12 = -1;
      if ( v11 )
      {
        _BitScanReverse64(&v11, v11);
        v12 = 63 - (v11 ^ 0x3F);
      }
    }
    v13 = (_QWORD *)sub_B2BE50(*a1);
    v14 = sub_BCCE00(v13, v12);
    v15 = sub_DC5200((__int64)a1, a2, v14, 0);
    return sub_DC2B70((__int64)a1, (__int64)v15, v9, 0);
  }
LABEL_11:
  v21[0] = sub_DCB270((__int64)a1, a2, a3);
  v20[0] = v21;
  v21[1] = a3;
  v20[1] = 0x200000002LL;
  v18 = sub_DC8BD0(a1, (__int64)v20, 2u, 0);
  if ( (_QWORD *)v20[0] != v21 )
    _libc_free(v20[0], v20);
  return sub_DCC810(a1, a2, (__int64)v18, 2, 0);
}
