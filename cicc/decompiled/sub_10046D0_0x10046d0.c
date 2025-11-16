// Function: sub_10046D0
// Address: 0x10046d0
//
__int64 __fastcall sub_10046D0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v8; // al
  __int64 v10; // r8
  size_t v11; // r15
  int v12; // eax
  __int64 v13; // rax
  __int64 v14; // [rsp+8h] [rbp-38h]

  v8 = *(_BYTE *)a2;
  if ( *(_BYTE *)a1 > 0x15u )
  {
    if ( v8 == 13 )
      return a1;
  }
  else if ( v8 <= 0x15u )
  {
    return sub_AAAE30(a1, a2, a3, a4);
  }
  if ( (unsigned __int8)sub_1003090(a5, (unsigned __int8 *)a2) && sub_98ED70((unsigned __int8 *)a1, 0, 0, 0, 0) )
    return a1;
  if ( *(_BYTE *)a2 == 93 )
  {
    v10 = *(_QWORD *)(a2 - 32);
    if ( *(_QWORD *)(v10 + 8) == *(_QWORD *)(a1 + 8) && *(_DWORD *)(a2 + 80) == a4 )
    {
      v11 = 4 * a4;
      if ( !v11 || (v14 = *(_QWORD *)(a2 - 32), v12 = memcmp(*(const void **)(a2 + 72), a3, v11), v10 = v14, !v12) )
      {
        if ( *(_BYTE *)a1 == 13 )
          return v10;
        if ( (unsigned __int8)sub_1003090(a5, (unsigned __int8 *)a1)
          && sub_98ED70(*(unsigned __int8 **)(a2 - 32), 0, 0, 0, 0) )
        {
          return *(_QWORD *)(a2 - 32);
        }
        v13 = *(_QWORD *)(a2 - 32);
        if ( a1 == v13 )
        {
          if ( v13 )
            return a1;
        }
      }
    }
  }
  return 0;
}
