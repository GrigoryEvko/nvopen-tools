// Function: sub_5D3C50
// Address: 0x5d3c50
//
__int64 __fastcall sub_5D3C50(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r12
  __int64 result; // rax
  char v4; // [rsp+7h] [rbp-29h] BYREF
  __int64 *v5; // [rsp+8h] [rbp-28h] BYREF

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 120);
  sub_72F9F0(a1, qword_4CF7E98, &v4, &v5);
  if ( v4 == 1 )
    v1 = *v5;
  if ( (unsigned int)sub_8D3410(v2) )
    v2 = sub_8D40F0(v2);
  while ( *(_BYTE *)(v2 + 140) == 12 )
    v2 = *(_QWORD *)(v2 + 160);
  if ( !(unsigned int)sub_8D3A70(v2) || (result = 1, (*(_BYTE *)(v2 + 176) & 8) == 0) )
  {
    result = 1;
    if ( (*(_BYTE *)(a1 + 173) & 8) == 0 )
      return (v4 == 2) & (unsigned __int8)(v1 == 0);
  }
  return result;
}
