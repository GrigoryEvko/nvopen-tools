// Function: sub_26EA740
// Address: 0x26ea740
//
__int64 __fastcall sub_26EA740(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // r8
  _QWORD *v3; // r9
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 56);
  v3 = *(_QWORD **)(*(_QWORD *)(a1 + 48) + 8 * (a2 % v2));
  if ( !v3 )
    return 0;
  v4 = (_QWORD *)*v3;
  if ( a2 != *(_QWORD *)(*v3 + 8LL) )
  {
    do
    {
      v5 = (_QWORD *)*v4;
      if ( !*v4 )
        return 0;
      v3 = v4;
      if ( a2 % v2 != v5[1] % v2 )
        return 0;
      v4 = (_QWORD *)*v4;
    }
    while ( a2 != v5[1] );
  }
  result = *v3;
  if ( *v3 )
    return *(unsigned int *)(result + 16);
  return result;
}
