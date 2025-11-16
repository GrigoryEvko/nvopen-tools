// Function: sub_14AD030
// Address: 0x14ad030
//
__int64 __fastcall sub_14AD030(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbp
  __int64 result; // rax
  __int64 v4; // [rsp-140h] [rbp-140h]
  __int64 v5; // [rsp-138h] [rbp-138h] BYREF
  _QWORD *v6; // [rsp-130h] [rbp-130h]
  _QWORD *v7; // [rsp-128h] [rbp-128h]
  __int64 v8; // [rsp-120h] [rbp-120h]
  int v9; // [rsp-118h] [rbp-118h]
  _QWORD v10[34]; // [rsp-110h] [rbp-110h] BYREF

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)a1 + 8LL) == 15 )
  {
    v10[33] = v2;
    v5 = 0;
    v6 = v10;
    v7 = v10;
    v8 = 32;
    v9 = 0;
    result = sub_14ACCB0(a1, (__int64)&v5, a2);
    if ( result == -1 )
      result = 1;
    if ( v7 != v6 )
    {
      v4 = result;
      _libc_free((unsigned __int64)v7);
      return v4;
    }
  }
  return result;
}
