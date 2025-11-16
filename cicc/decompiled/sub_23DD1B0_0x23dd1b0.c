// Function: sub_23DD1B0
// Address: 0x23dd1b0
//
_QWORD *__fastcall sub_23DD1B0(__int64 a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  _QWORD *v4; // r14
  char v5; // r15
  _QWORD *v6; // r12
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v10; // [rsp+8h] [rbp-68h]
  _QWORD v11[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v12; // [rsp+30h] [rbp-40h]

  v4 = *(_QWORD **)(a2 + 8);
  v5 = (*(_DWORD *)(a1 + 100) != 5) + 7;
  if ( a4 && *a3 == 1 )
  {
    --a4;
    ++a3;
  }
  v11[2] = a3;
  v11[0] = "__asan_global_";
  v12 = 1283;
  v11[3] = a4;
  BYTE4(v10) = 0;
  v6 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v6 )
    sub_B30000((__int64)v6, *(_QWORD *)a1, v4, 0, v5, a2, (__int64)v11, 0, 0, v10, 0);
  v7 = sub_23DBFF0(a1);
  sub_B31A00((__int64)v6, (__int64)v7, v8);
  sub_29F3D50(a1 + 48, v6);
  return v6;
}
