// Function: sub_E31E90
// Address: 0xe31e90
//
__int64 __fastcall sub_E31E90(__int64 a1, char a2)
{
  unsigned __int64 v2; // rax
  unsigned __int64 v3; // rdx
  __int64 v4; // r8
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rdx
  __int64 result; // rax

  if ( *(_BYTE *)(a1 + 49) )
    return 0;
  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 24);
  if ( v2 >= v3 )
    return 0;
  v4 = *(_QWORD *)(a1 + 32);
  if ( a2 != *(_BYTE *)(v4 + v2) )
    return 0;
  *(_QWORD *)(a1 + 40) = v2 + 1;
  if ( v3 > v2 + 1 && *(_BYTE *)(v4 + v2 + 1) == 95 )
  {
    *(_QWORD *)(a1 + 40) = v2 + 2;
    return 1;
  }
  v5 = sub_E31BC0(a1);
  v6 = v5;
  if ( !*(_BYTE *)(a1 + 49) )
  {
    result = v5 + 1;
    if ( v6 != -1 )
      return result;
    *(_BYTE *)(a1 + 49) = 1;
  }
  return 0;
}
