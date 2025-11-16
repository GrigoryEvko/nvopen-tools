// Function: sub_98B430
// Address: 0x98b430
//
__int64 __fastcall sub_98B430(__int64 a1, unsigned int a2)
{
  __int64 v2; // rbp
  __int64 result; // rax
  __int64 v4; // [rsp-130h] [rbp-130h]
  __int64 v5; // [rsp-128h] [rbp-128h] BYREF
  _QWORD *v6; // [rsp-120h] [rbp-120h]
  __int64 v7; // [rsp-118h] [rbp-118h]
  int v8; // [rsp-110h] [rbp-110h]
  char v9; // [rsp-10Ch] [rbp-10Ch]
  _QWORD v10[33]; // [rsp-108h] [rbp-108h] BYREF

  result = 0;
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 14 )
  {
    v10[32] = v2;
    v5 = 0;
    v6 = v10;
    v7 = 32;
    v8 = 0;
    v9 = 1;
    result = sub_98B210(a1, (__int64)&v5, a2);
    if ( result == -1 )
      result = 1;
    if ( !v9 )
    {
      v4 = result;
      _libc_free(v6, &v5);
      return v4;
    }
  }
  return result;
}
