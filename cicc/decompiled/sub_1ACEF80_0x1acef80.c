// Function: sub_1ACEF80
// Address: 0x1acef80
//
__int64 __fastcall sub_1ACEF80(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v5; // r12
  int v6; // eax
  __int64 v7; // rax
  _QWORD v8[3]; // [rsp+0h] [rbp-A0h] BYREF
  bool v9; // [rsp+18h] [rbp-88h]
  __int64 v10; // [rsp+20h] [rbp-80h]
  _BYTE *v11; // [rsp+28h] [rbp-78h]
  _BYTE *v12; // [rsp+30h] [rbp-70h]
  __int64 v13; // [rsp+38h] [rbp-68h]
  int v14; // [rsp+40h] [rbp-60h]
  _BYTE v15[88]; // [rsp+48h] [rbp-58h] BYREF

  v8[0] = a1;
  v8[1] = a2;
  v8[2] = a3;
  v9 = 0;
  v10 = 0;
  v11 = v15;
  v12 = v15;
  v13 = 8;
  v14 = 0;
  if ( !a3 )
  {
    v5 = *(_QWORD *)(a2 + 48) + 8LL * *(unsigned int *)(a2 + 56);
    v6 = sub_16D1B30((__int64 *)(a2 + 48), *(unsigned __int8 **)(a1 + 176), *(_QWORD *)(a1 + 184));
    if ( v6 == -1 )
      v7 = *(_QWORD *)(a2 + 48) + 8LL * *(unsigned int *)(a2 + 56);
    else
      v7 = *(_QWORD *)(a2 + 48) + 8LL * v6;
    v9 = v7 != v5;
  }
  v3 = sub_1ACEF70((__int64)v8);
  if ( v12 != v11 )
    _libc_free((unsigned __int64)v12);
  return v3;
}
