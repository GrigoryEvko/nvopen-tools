// Function: sub_1875960
// Address: 0x1875960
//
_QWORD *__fastcall sub_1875960(__int64 **a1, _QWORD **a2, __int64 *a3, __int64 *a4)
{
  __int64 *v5; // rax
  __int64 *v6; // rax
  __int64 v7; // rax
  __int64 *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  _QWORD *result; // rax
  _QWORD **v13; // [rsp+0h] [rbp-70h] BYREF
  __int16 v14; // [rsp+10h] [rbp-60h]
  _QWORD *v15; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v16[2]; // [rsp+30h] [rbp-40h] BYREF
  int v17; // [rsp+40h] [rbp-30h]
  int v18; // [rsp+4Ch] [rbp-24h]
  int v19; // [rsp+54h] [rbp-1Ch]

  a1[2] = a4;
  a1[1] = a3;
  *a1 = (__int64 *)a2;
  a1[5] = (__int64 *)sub_1643320(*a2);
  a1[6] = (__int64 *)sub_1643330((_QWORD *)**a1);
  a1[7] = (__int64 *)sub_16471D0((_QWORD *)**a1, 0);
  v5 = (__int64 *)sub_1643330((_QWORD *)**a1);
  a1[8] = sub_1645D80(v5, 0);
  v6 = (__int64 *)sub_1643350((_QWORD *)**a1);
  a1[9] = v6;
  a1[10] = (__int64 *)sub_1646BA0(v6, 0);
  v7 = sub_1643360((_QWORD *)**a1);
  v8 = *a1;
  a1[11] = (__int64 *)v7;
  v9 = sub_1632FA0((__int64)v8);
  v10 = sub_15A9620(v9, **a1, 0);
  a1[13] = (__int64 *)1;
  a1[12] = (__int64 *)v10;
  a1[14] = 0;
  a1[15] = 0;
  a1[16] = 0;
  *((_DWORD *)a1 + 34) = 0;
  a1[18] = 0;
  a1[19] = 0;
  a1[20] = 0;
  a1[21] = 0;
  v14 = 260;
  v13 = a2 + 30;
  sub_16E1010((__int64)&v15, (__int64)&v13);
  v11 = v15;
  *((_DWORD *)a1 + 6) = v17;
  *((_DWORD *)a1 + 7) = v18;
  *((_DWORD *)a1 + 8) = v19;
  result = v16;
  if ( v11 != v16 )
    return (_QWORD *)j_j___libc_free_0(v11, v16[0] + 1LL);
  return result;
}
