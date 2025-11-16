// Function: sub_22F3E50
// Address: 0x22f3e50
//
__int64 __fastcall sub_22F3E50(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rdx
  _QWORD v5[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v6[4]; // [rsp+10h] [rbp-20h] BYREF

  v5[0] = sub_22F59B0(a1[1], *(unsigned __int16 *)(*a1 + 58LL));
  v5[1] = v2;
  if ( v5[0] )
    return sub_22F3E50(v5, a2);
  result = 1;
  if ( a2 != *(_DWORD *)(*a1 + 40LL) )
  {
    v6[0] = sub_22F59B0(a1[1], *(unsigned __int16 *)(*a1 + 56LL));
    result = 0;
    v6[1] = v4;
    if ( v6[0] )
      return sub_22F3E50(v6, a2);
  }
  return result;
}
