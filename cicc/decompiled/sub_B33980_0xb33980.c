// Function: sub_B33980
// Address: 0xb33980
//
__int64 __fastcall sub_B33980(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rsi
  _QWORD v9[6]; // [rsp-30h] [rbp-30h] BYREF

  result = *(_QWORD *)a1;
  v3 = *(_QWORD *)a1 + 16LL * *(unsigned int *)(a1 + 8);
  if ( v3 != *(_QWORD *)a1 )
  {
    while ( *(_DWORD *)result )
    {
      result += 16;
      if ( v3 == result )
        return result;
    }
    result = sub_B10CB0(v9, *(_QWORD *)(result + 8));
    if ( *(_QWORD *)(a2 + 48) )
      result = sub_B91220(a2 + 48);
    v8 = v9[0];
    *(_QWORD *)(a2 + 48) = v9[0];
    if ( v8 )
      return sub_B976B0(v9, v8, a2 + 48, v5, v6, v7);
  }
  return result;
}
