// Function: sub_1530BF0
// Address: 0x1530bf0
//
__int64 __fastcall sub_1530BF0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 i; // rsi
  __int64 v6; // rax
  __int64 v7; // rdi

  result = *(_QWORD *)(a1 + 32);
  if ( *(_QWORD *)(a1 + 24) != result && *(_QWORD *)(result - 32) == a2 )
  {
    sub_1526BE0(*(_QWORD **)a1, 0x12u, 3u);
    for ( i = *(_QWORD *)(a1 + 32); i != *(_QWORD *)(a1 + 24); i = *(_QWORD *)(a1 + 32) )
    {
      if ( *(_QWORD *)(i - 32) != a2 )
        break;
      sub_1530A50((_DWORD **)a1, (__int64 *)(i - 40));
      v6 = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(a1 + 32) = v6 - 40;
      v7 = *(_QWORD *)(v6 - 24);
      if ( v7 )
        j_j___libc_free_0(v7, *(_QWORD *)(v6 - 8) - v7);
    }
    return sub_15263C0(*(__int64 ***)a1);
  }
  return result;
}
