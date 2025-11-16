// Function: sub_DD99C0
// Address: 0xdd99c0
//
__int64 *__fastcall sub_DD99C0(__int64 *a1, __int64 a2, __int64 a3, _BYTE *a4, _BYTE *a5)
{
  _BYTE *v6; // r13
  unsigned int v8; // ebx
  __int64 *result; // rax
  __int64 v10; // r9
  char v11; // dl

  v6 = a5;
  if ( *(_BYTE *)a3 == 17 )
  {
    v8 = *(_DWORD *)(a3 + 32);
    if ( v8 <= 0x40 )
    {
      if ( *(_QWORD *)(a3 + 24) == 1 )
        v6 = a4;
    }
    else if ( (unsigned int)sub_C444A0(a3 + 24) == v8 - 1 )
    {
      v6 = a4;
    }
    return sub_DD8400((__int64)a1, (__int64)v6);
  }
  else
  {
    v10 = a2;
    if ( *(_BYTE *)a2 <= 0x1Cu )
      return sub_DD8D60(a1, v10, a3, a4, v6);
    if ( *(_BYTE *)a3 != 82 )
      return sub_DD8D60(a1, v10, a3, a4, v6);
    result = sub_DD9390(a1, *(_QWORD *)(a2 + 8), a3, a4, a5);
    v10 = a2;
    if ( !v11 )
      return sub_DD8D60(a1, v10, a3, a4, v6);
  }
  return result;
}
