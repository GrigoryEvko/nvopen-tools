// Function: sub_120C270
// Address: 0x120c270
//
__int64 __fastcall sub_120C270(__int64 a1, _QWORD **a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  int v5; // eax
  size_t v6; // r8
  _QWORD *v7; // rcx
  unsigned int v8; // r12d
  const void *v10[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD v11[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD *v12; // [rsp+30h] [rbp-50h] BYREF
  size_t v13; // [rsp+38h] [rbp-48h]
  _QWORD v14[8]; // [rsp+40h] [rbp-40h] BYREF

  v3 = *(_BYTE **)(a1 + 248);
  v4 = *(_QWORD *)(a1 + 256);
  v10[0] = v11;
  sub_12060D0((__int64 *)v10, v3, (__int64)&v3[v4]);
  v5 = sub_1205200(a1 + 176);
  v12 = v14;
  v6 = 0;
  v7 = v14;
  *(_DWORD *)(a1 + 240) = v5;
  v13 = 0;
  LOBYTE(v14[0]) = 0;
  if ( v5 != 3 )
    goto LABEL_2;
  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  v8 = sub_120B3D0(a1, (__int64)&v12);
  if ( !(_BYTE)v8 )
  {
    v7 = v12;
    v6 = v13;
LABEL_2:
    v8 = 0;
    sub_A78980(a2, v10[0], (size_t)v10[1], v7, v6);
  }
  if ( v12 != v14 )
    j_j___libc_free_0(v12, v14[0] + 1LL);
  if ( v10[0] != v11 )
    j_j___libc_free_0(v10[0], v11[0] + 1LL);
  return v8;
}
