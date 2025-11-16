// Function: sub_145F6E0
// Address: 0x145f6e0
//
__int64 __fastcall sub_145F6E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned int v4; // r12d
  __int64 v6; // r14
  unsigned int v7; // r12d
  __int64 v8; // rcx
  unsigned int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rsi
  int v13; // r10d
  __int64 v14; // r8
  void *v15; // [rsp+0h] [rbp-80h] BYREF
  _BYTE v16[16]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  void *v18; // [rsp+30h] [rbp-50h] BYREF
  _BYTE v19[16]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v20; // [rsp+48h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return v4;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v4 - 1;
  sub_1457D90(&v15, -8, 0);
  sub_1457D90(&v18, -16, 0);
  v8 = *(_QWORD *)(a2 + 24);
  v9 = v7 & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
  v10 = v6 + 48LL * v9;
  v11 = *(_QWORD *)(v10 + 24);
  if ( v8 == v11 )
  {
    v12 = v20;
LABEL_6:
    *a3 = v10;
    v4 = 1;
  }
  else
  {
    v12 = v20;
    v13 = 1;
    v14 = 0;
    while ( v17 != v11 )
    {
      if ( !v14 && v20 == v11 )
        v14 = v10;
      v9 = v7 & (v13 + v9);
      v10 = v6 + 48LL * v9;
      v11 = *(_QWORD *)(v10 + 24);
      if ( v11 == v8 )
        goto LABEL_6;
      ++v13;
    }
    if ( !v14 )
      v14 = v10;
    v4 = 0;
    *a3 = v14;
  }
  v18 = &unk_49EE2B0;
  if ( v12 != -8 && v12 != 0 && v12 != -16 )
    sub_1649B30(v19);
  v15 = &unk_49EE2B0;
  if ( v17 == 0 || v17 == -8 || v17 == -16 )
    return v4;
  sub_1649B30(v16);
  return v4;
}
