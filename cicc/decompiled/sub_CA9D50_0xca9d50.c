// Function: sub_CA9D50
// Address: 0xca9d50
//
__int64 __fastcall sub_CA9D50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        const char *a9,
        const char *a10)
{
  __int64 v10; // rax
  __int64 *v11; // r12
  _QWORD *v12; // rdi
  __int64 result; // rax
  __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 v15[8]; // [rsp+10h] [rbp-40h] BYREF

  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 257;
  *(_QWORD *)(a1 + 16) = a8;
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(a1 + 16) + a7;
  *(_QWORD *)(a1 + 8) = a7;
  *(_QWORD *)(a1 + 40) = a7;
  *(_QWORD *)(a1 + 56) = 0xFFFFFFFFLL;
  *(_QWORD *)(a1 + 24) = a9;
  *(_QWORD *)(a1 + 32) = a10;
  sub_C7DB10(&v14, 0, a3, a4, (__int64)a9, (__int64)a10, a7, a8, a9, a10);
  v10 = v14;
  v11 = *(__int64 **)a1;
  v15[1] = 0;
  v15[2] = 0;
  v12 = (_QWORD *)v11[1];
  v14 = 0;
  v15[0] = v10;
  if ( v12 == (_QWORD *)v11[2] )
  {
    sub_C12520(v11, (__int64)v12, (__int64)v15);
  }
  else
  {
    if ( v12 )
    {
      sub_C8EDF0(v12, v15);
      v12 = (_QWORD *)v11[1];
    }
    v11[1] = (__int64)(v12 + 3);
  }
  result = sub_C8EE20(v15);
  if ( v14 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 8LL))(v14);
  return result;
}
