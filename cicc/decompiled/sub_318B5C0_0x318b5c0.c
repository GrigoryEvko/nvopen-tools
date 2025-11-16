// Function: sub_318B5C0
// Address: 0x318b5c0
//
__int64 __fastcall sub_318B5C0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 result; // rax
  __int64 v5; // rdx

  v1 = sub_318B520(a1);
  if ( v1 )
  {
    v2 = *(_QWORD *)(v1 + 16);
    v3 = *(_QWORD *)(v2 + 32);
    if ( v3 != *(_QWORD *)(v2 + 40) + 48LL && v3 )
      return v3 - 24;
    return 0;
  }
  v5 = *(_QWORD *)(*(_QWORD *)(sub_318B4F0(a1) + 16) + 56LL);
  result = v5 - 24;
  if ( !v5 )
    return 0;
  return result;
}
