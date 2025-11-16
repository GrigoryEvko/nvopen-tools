// Function: sub_3223AD0
// Address: 0x3223ad0
//
__int64 __fastcall sub_3223AD0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // r11
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r8
  _QWORD *v6; // r9
  _QWORD *v7; // rax
  unsigned __int64 v8; // rcx
  __int64 result; // rax

  v2 = a2[1];
  v4 = a1[1];
  v5 = v2 + 31LL * *a2;
  v6 = *(_QWORD **)(*a1 + 8 * (v5 % v4));
  if ( !v6 )
    return 0;
  v7 = (_QWORD *)*v6;
  v8 = *(_QWORD *)(*v6 + 208LL);
  while ( v5 != v8 || *a2 != v7[1] || v2 != v7[2] )
  {
    if ( !*v7 )
      return 0;
    v8 = *(_QWORD *)(*v7 + 208LL);
    v6 = v7;
    if ( v5 % v4 != v8 % v4 )
      return 0;
    v7 = (_QWORD *)*v7;
  }
  result = *v6;
  if ( *v6 )
    return result;
  return 0;
}
