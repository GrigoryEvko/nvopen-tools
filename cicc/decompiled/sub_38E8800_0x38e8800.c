// Function: sub_38E8800
// Address: 0x38e8800
//
__int64 __fastcall sub_38E8800(__int64 a1, _BYTE *a2, __int64 a3, char a4, char a5)
{
  unsigned int v6; // r12d
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_38E8410(a2, a3, a4, (_QWORD *)a1, &v8, v9);
  if ( !(_BYTE)v6 )
  {
    if ( v8 )
    {
      (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 240LL))(
        *(_QWORD *)(a1 + 328),
        v8,
        v9[0]);
      if ( a5 )
        (*(void (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 328) + 256LL))(
          *(_QWORD *)(a1 + 328),
          v8,
          14);
    }
  }
  return v6;
}
