// Function: sub_3268F30
// Address: 0x3268f30
//
__int64 __fastcall sub_3268F30(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 result; // rax
  __int64 v9; // rsi
  __int64 v10; // rbx
  __int64 *v11; // rdi
  __int64 v12; // [rsp-E0h] [rbp-E0h]
  __int64 v13; // [rsp-E0h] [rbp-E0h]
  __int64 v14; // [rsp-D8h] [rbp-D8h] BYREF
  int v15; // [rsp-D0h] [rbp-D0h]
  __int64 *v16; // [rsp-C8h] [rbp-C8h] BYREF
  __int64 v17; // [rsp-C0h] [rbp-C0h]
  _BYTE v18[184]; // [rsp-B8h] [rbp-B8h] BYREF

  if ( !*((_DWORD *)a1 + 7) )
    return a3;
  v17 = 0x800000000LL;
  v16 = (__int64 *)v18;
  sub_3268A80(a1, a2, a3, a4, (__int64)&v16, a6);
  if ( (_DWORD)v17 )
  {
    if ( (_DWORD)v17 == 1 )
    {
      v11 = v16;
      result = *v16;
    }
    else
    {
      v9 = *(_QWORD *)(a2 + 80);
      v10 = *a1;
      v14 = v9;
      if ( v9 )
        sub_B96E90((__int64)&v14, v9, 1);
      v15 = *(_DWORD *)(a2 + 72);
      result = sub_3402E70(v10, &v14, &v16);
      if ( v14 )
      {
        v12 = result;
        sub_B91220((__int64)&v14, v14);
        result = v12;
      }
      v11 = v16;
    }
  }
  else
  {
    v11 = v16;
    result = *a1 + 288;
  }
  if ( v11 != (__int64 *)v18 )
  {
    v13 = result;
    _libc_free((unsigned __int64)v11);
    return v13;
  }
  return result;
}
