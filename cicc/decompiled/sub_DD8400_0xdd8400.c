// Function: sub_DD8400
// Address: 0xdd8400
//
__int64 *__fastcall sub_DD8400(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  unsigned int v4; // eax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rsi
  unsigned int v8; // edx
  unsigned int v9; // eax
  bool v10; // cc
  __int64 *v11; // [rsp+8h] [rbp-18h]

  result = (__int64 *)sub_D98300(a1, a2);
  if ( result )
    return result;
  if ( !*(_BYTE *)(a1 + 1560) )
  {
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 40LL) + 264LL) - 42) <= 1 )
      return (__int64 *)sub_DD8130((_QWORD *)a1, a2);
    v9 = *(_DWORD *)(a1 + 1568) + 1;
    v10 = v9 <= dword_4F88268;
    *(_DWORD *)(a1 + 1568) = v9;
    if ( v10 )
      return (__int64 *)sub_DD8130((_QWORD *)a1, a2);
    return sub_DA3860((_QWORD *)a1, a2);
  }
  if ( *(_DWORD *)(a1 + 1564) > (unsigned int)qword_4F88348 )
    return sub_DA3860((_QWORD *)a1, a2);
  v4 = 1;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v5 = *(_QWORD *)(a2 + 40);
    v6 = *(_QWORD *)(a1 + 40);
    if ( v5 )
    {
      v7 = (unsigned int)(*(_DWORD *)(v5 + 44) + 1);
      v8 = *(_DWORD *)(v5 + 44) + 1;
    }
    else
    {
      v7 = 0;
      v8 = 0;
    }
    v4 = 1;
    if ( v8 < *(_DWORD *)(v6 + 32) && *(_QWORD *)(*(_QWORD *)(v6 + 24) + 8 * v7) )
      v4 = sub_DB3670(a2, *(_QWORD *)(a1 + 48), *(_BYTE *)(a1 + 1572));
  }
  if ( dword_4F88428 >= v4 )
  {
    *(_BYTE *)(a1 + 1560) = 0;
    *(_DWORD *)(a1 + 1568) = 0;
    result = (__int64 *)sub_DD8130((_QWORD *)a1, a2);
    *(_BYTE *)(a1 + 1560) = 1;
  }
  else
  {
    ++*(_DWORD *)(a1 + 1564);
    v11 = sub_DA3860((_QWORD *)a1, a2);
    sub_DB77A0(a1, a2, (__int64)v11);
    return v11;
  }
  return result;
}
