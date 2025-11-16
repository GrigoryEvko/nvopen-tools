// Function: sub_122E830
// Address: 0x122e830
//
__int64 __fastcall sub_122E830(__int64 a1, _DWORD *a2, __int64 *a3)
{
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  unsigned int v7; // r12d
  const void *v9[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v10[6]; // [rsp+10h] [rbp-30h] BYREF

  v5 = *(_BYTE **)(a1 + 248);
  v6 = *(_QWORD *)(a1 + 256);
  v9[0] = v10;
  sub_12060D0((__int64 *)v9, v5, (__int64)&v5[v6]);
  *a2 = sub_BA8BE0(*(__int64 ***)(a1 + 344), v9[0], (size_t)v9[1]);
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v7 = sub_122E7D0(a1, a3);
  if ( v9[0] != v10 )
    j_j___libc_free_0(v9[0], v10[0] + 1LL);
  return v7;
}
