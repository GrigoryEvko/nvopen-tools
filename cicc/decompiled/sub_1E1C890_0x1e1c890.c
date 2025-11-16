// Function: sub_1E1C890
// Address: 0x1e1c890
//
__int64 __fastcall sub_1E1C890(__int64 a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r14
  __int64 v5; // rcx
  __int64 v6; // r15

  v3 = (_QWORD *)*a3;
  v4 = (_QWORD *)a3[1];
  if ( (_QWORD *)*a3 == v4 )
    return 0;
  while ( 1 )
  {
    v5 = 0;
    v6 = *v3;
    if ( *(_BYTE *)(a1 + 552) )
      v5 = *(_QWORD *)(a1 + 264);
    if ( (*(unsigned __int8 (__fastcall **)(_QWORD, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 184LL))(
           *(_QWORD *)(a1 + 232),
           a2,
           *v3,
           v5) )
    {
      break;
    }
    if ( v4 == ++v3 )
      return 0;
  }
  return v6;
}
