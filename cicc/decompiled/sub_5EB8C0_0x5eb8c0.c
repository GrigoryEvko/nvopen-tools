// Function: sub_5EB8C0
// Address: 0x5eb8c0
//
_BOOL8 __fastcall sub_5EB8C0(__int64 a1)
{
  __int64 i; // rax
  _QWORD *v2; // rdx
  _BYTE *v3; // rax
  _BOOL8 result; // rax
  int v5; // r8d

  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v2 = *(_QWORD **)(i + 168);
  v3 = (_BYTE *)v2[7];
  if ( !v3 || (*v3 & 8) != 0 || !*v2 || !(unsigned int)sub_8D3110(*(_QWORD *)(*v2 + 8LL)) )
    return 0;
  v5 = sub_72F570(a1);
  result = 1;
  if ( !v5 )
    return (unsigned int)sub_72F850(a1) != 0;
  return result;
}
