// Function: sub_89EDD0
// Address: 0x89edd0
//
_BOOL8 __fastcall sub_89EDD0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 i; // rax
  bool v4; // zf
  _BOOL8 result; // rax

  *a3 = 0;
  for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)i + 96LL) + 72LL) == a2;
  result = v4;
  if ( v4 )
    *a3 = *(_QWORD *)(*(_QWORD *)(a1 + 168) + 168LL);
  return result;
}
