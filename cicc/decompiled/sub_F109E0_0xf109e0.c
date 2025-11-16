// Function: sub_F109E0
// Address: 0xf109e0
//
__int64 __fastcall sub_F109E0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rdi
  __int64 result; // rax
  bool v7; // zf

  v3 = (_QWORD *)(a2 + 24);
  v4 = *(_QWORD **)(*(_QWORD *)(a2 + 40) + 56LL);
  do
  {
    if ( v4 == v3 )
      break;
    v3 = (_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v4 == v3 )
      break;
    v5 = (__int64)(v3 - 3);
    if ( !v3 )
      v5 = 0;
  }
  while ( sub_B46AA0(v5) );
  if ( !v3 )
    BUG();
  result = 0;
  if ( *((_BYTE *)v3 - 24) == 62 )
  {
    v7 = (unsigned __int8)sub_114FFE0(a1, v3 - 3) == 0;
    result = 0;
    if ( !v7 )
      return a2;
  }
  return result;
}
