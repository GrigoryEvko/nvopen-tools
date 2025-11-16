// Function: sub_37C6A40
// Address: 0x37c6a40
//
__int64 __fastcall sub_37C6A40(_QWORD *a1, __int64 a2, __int64 a3)
{
  bool v5; // zf
  __int64 v6; // r14
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r14

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] || !(unsigned __int8)sub_37B9A70(a1[4] + 32LL, (int *)a3) )
      return sub_37C6950((__int64)a1, (int *)a3);
    return 0;
  }
  v5 = *(_DWORD *)(a2 + 32) == *(_DWORD *)a3;
  if ( *(_DWORD *)(a2 + 32) > *(_DWORD *)a3
    || (v8 = *(_QWORD *)(a2 + 48), v9 = *(_QWORD *)(a2 + 40), v10 = *(_QWORD *)(a3 + 8), v5)
    && (v9 > v10 || v9 == v10 && v8 > *(_QWORD *)(a3 + 16)) )
  {
    if ( a1[3] == a2 )
      return a2;
    v6 = sub_220EF80(a2);
    if ( (unsigned __int8)sub_37B9A70(v6 + 32, (int *)a3) )
    {
      result = 0;
      if ( *(_QWORD *)(v6 + 24) )
        return a2;
      return result;
    }
    return sub_37C6950((__int64)a1, (int *)a3);
  }
  if ( !(unsigned __int8)sub_37B9A70(a2 + 32, (int *)a3) )
    return a2;
  if ( a1[4] == a2 )
    return 0;
  v11 = sub_220EEE0(a2);
  if ( !(unsigned __int8)sub_37B9A70(a3, (int *)(v11 + 32)) )
    return sub_37C6950((__int64)a1, (int *)a3);
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v11;
  return result;
}
