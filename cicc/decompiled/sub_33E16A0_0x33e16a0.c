// Function: sub_33E16A0
// Address: 0x33e16a0
//
__int64 __fastcall sub_33E16A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // edx
  __int64 result; // rax
  char v9; // r14
  __int64 v10; // rax
  _DWORD *v11; // r8
  __int64 v12; // r12
  _QWORD *v13; // rax
  _QWORD *v14; // rcx
  int v15; // edx
  _QWORD *v16; // [rsp-78h] [rbp-78h] BYREF
  __int64 v17; // [rsp-70h] [rbp-70h]
  _DWORD v18[26]; // [rsp-68h] [rbp-68h] BYREF

  v7 = *(_DWORD *)(a1 + 24);
  result = a1;
  if ( v7 == 12 || v7 == 36 )
    return result;
  if ( v7 != 156 )
    goto LABEL_10;
  v9 = a4;
  v16 = v18;
  v17 = 0x600000000LL;
  v18[12] = 0;
  v10 = sub_33E1670(a1, a3, (__int64)&v16, a4, a5, a6);
  v11 = v16;
  v12 = v10;
  if ( !v10 || (v13 = sub_33C7FB0(v16, (__int64)&v16[(unsigned int)v17]), v14 != v13) && !v9 )
  {
    if ( v11 != v18 )
      _libc_free((unsigned __int64)v11);
    v7 = *(_DWORD *)(a1 + 24);
LABEL_10:
    result = 0;
    if ( v7 == 168 )
    {
      result = **(_QWORD **)(a1 + 40);
      v15 = *(_DWORD *)(result + 24);
      if ( v15 != 12 && v15 != 36 )
        return 0;
    }
    return result;
  }
  if ( v11 != v18 )
    _libc_free((unsigned __int64)v11);
  return v12;
}
