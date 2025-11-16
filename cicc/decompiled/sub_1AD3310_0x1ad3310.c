// Function: sub_1AD3310
// Address: 0x1ad3310
//
__int64 __fastcall sub_1AD3310(__int64 *a1)
{
  _QWORD **v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 result; // rax

  v1 = (_QWORD **)*a1;
  v2 = *a1;
  if ( *(_BYTE *)(*a1 + 8) == 16 )
    v2 = *v1[2];
  v3 = sub_16471D0(*v1, *(_DWORD *)(v2 + 8) >> 8);
  if ( v1 == (_QWORD **)v3 )
    return sub_1AD32A0(a1[1]);
  v4 = a1[1];
  if ( !v4 )
    return 0;
  while ( 1 )
  {
    v5 = sub_1648700(v4);
    v6 = v5;
    if ( v3 == *v5 && a1 == (__int64 *)sub_1649C60((__int64)v5) )
    {
      result = sub_1AD32A0(v6[1]);
      if ( (_BYTE)result )
        break;
    }
    v4 = *(_QWORD *)(v4 + 8);
    if ( !v4 )
      return 0;
  }
  return result;
}
